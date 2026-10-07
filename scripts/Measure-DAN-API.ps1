param(
    [string]$Endpoint = 'http://127.0.0.1:8080/v1/chat/completions',
    [int]$Repeats = 3,
    [int[]]$BackgroundWords = @(0, 2000),
    [string]$OutputFile
)
$ErrorActionPreference = 'Stop'
$http = [Net.Http.HttpClient]::new()
$http.Timeout = [TimeSpan]::FromMinutes(5)
try {
    $results = foreach ($words in $BackgroundWords) {
        for ($run = 1; $run -le $Repeats; $run++) {
            $prompt = ('background ' * $words) + ' Explain GPU computing in detail, covering memory, parallelism, and inference.'
            $body = @{model='dan-auto'; stream=$true; max_tokens=128; messages=@(@{role='user';content=$prompt})} | ConvertTo-Json -Depth 5 -Compress
            $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, $Endpoint)
            $request.Content = [Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
            $timer = [Diagnostics.Stopwatch]::StartNew()
            $response = $null
            $reader = $null
            try {
                $response = $http.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
                $response.EnsureSuccessStatusCode() | Out-Null
                $reader = [IO.StreamReader]::new($response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
                $first = $null
                $events = 0
                $done = $false
                while ($null -ne ($line = $reader.ReadLine())) {
                    if ($line -eq 'data: [DONE]') { $done = $true; break }
                    if (-not $line.StartsWith('data: ')) { continue }
                    $event = $line.Substring(6) | ConvertFrom-Json
                    if ($event.error) { throw $event.error.message }
                    if ($event.choices[0].delta.content) {
                        if ($null -eq $first) { $first = $timer.Elapsed.TotalSeconds }
                        $events++
                    }
                }
                if (-not $done -or $null -eq $first) { throw 'Incomplete streaming response' }
                [pscustomobject]@{
                    background_words=$words; run=$run
                    first_text_seconds=[Math]::Round($first, 3)
                    total_seconds=[Math]::Round($timer.Elapsed.TotalSeconds, 3)
                    content_events=$events
                }
            } finally {
                if ($reader) { $reader.Dispose() }
                if ($response) { $response.Dispose() }
                $request.Dispose()
            }
        }
    }
    # Content events are transport chunks, not a guaranteed token count.
    if ($OutputFile) { $results | ConvertTo-Json | Set-Content -LiteralPath $OutputFile -Encoding UTF8 }
    $results
} finally { $http.Dispose() }
