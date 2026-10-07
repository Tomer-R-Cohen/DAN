"""Check schema output, RAM reuse and cancellation against a running local DAN API.

Uses the configured real model; no UI, timing assertions or conversation files.
"""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    args = parser.parse_args()

    def request(body):
        req = urllib.request.Request(args.url, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'X-OpenWebUI-User-Id': 'correctness-check'})
        return urllib.request.urlopen(req, timeout=90)

    body = {'model': 'dan-auto', 'messages': [{'role': 'user', 'content': 'Return the required JSON object.'}],
        'temperature': 0.7, 'seed': 123, 'max_tokens': 32, 'prompt_cache_key': 'schema-check',
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'check', 'strict': True,
            'schema': {'type': 'object', 'properties': {'answer': {'const': 4}},
                'required': ['answer'], 'additionalProperties': False}}}}
    for repeated in (False, True):
        with request(body) as response:
            result = json.load(response)
        assert json.loads(result['choices'][0]['message']['content']) == {'answer': 4}
        if repeated: assert result['usage']['prompt_tokens_details']['cached_tokens'] > 0
    print('PASS replica structured output and prefix reuse')

    with request({'model': 'dan-auto', 'messages': [{'role': 'user',
            'content': 'List integers from 1 to 500, one per line.'}],
            'stream': True, 'max_tokens': 2048, 'prompt_cache_key': 'cancel-check'}) as response:
        for line in response:
            if b'"content"' in line: break
        else: raise AssertionError('stream ended without content')
    # Closing the stream must cancel and release the only replica slot before reuse.
    body['prompt_cache_key'] = 'after-cancel'
    with request(body) as response:
        result = json.load(response)
    assert json.loads(result['choices'][0]['message']['content']) == {'answer': 4}
    print('PASS cancelled stream releases session; next request succeeds')


if __name__ == '__main__':
    main()
