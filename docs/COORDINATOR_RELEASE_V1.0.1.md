DAN Coordinator v1.0.1 - Windows x64
====================================

Requirements
------------

- Windows 10 or 11, 64-bit
- At least two available DAN providers for the bundled encrypted ring

Dependency licenses and any required reciprocal source are included under
`licenses` in the extracted package.
Model weights are downloaded separately and are not inside the archive. The
bundled Qwen2.5 configurations identify exact source revisions; those revisions
declare Apache-2.0, whose model license is included as `licenses\Qwen2.5-LICENSE.txt`.

Start
-----

1. Extract the entire ZIP.
2. Double-click `dan-coordinator.exe`.
3. Allow network access if Windows Firewall asks.
4. DAN creates a permanent coordinator identity and prints its connection addresses.
5. Put one reachable printed address into the provider package before publishing it.
6. Wait while DAN selects providers and they download their assigned model ranges.
7. When `REPLICA READY` appears, type directly into the coordinator window.

Commands: `/new` resets the conversation, `/stats` shows current totals, and
`/quit` stops the coordinator. Providers reconnect automatically afterward.

Model selection
---------------

The default is Qwen2.5 0.5B Q4_K_M. Other tested Qwen2 model configurations are
in `config\models`. To switch, stop the coordinator and copy the selected JSON
over `config\active-model.json`, then start it again. The coordinator downloads
metadata only; providers download their assigned weight ranges.

The bundled peer mode uses at least two providers and divides the model in
proportion to their contributed VRAM. Direct/manual mode may use one provider.

API service
-----------

Double-click `Start-DAN-Service.cmd` to start the encrypted provider network,
coordinator, and DAN API together. It listens on loopback by default. A
non-loopback listen is rejected unless a strong `DAN_API_KEY` is set; expose it
only through a TLS reverse proxy:

```powershell
$env:DAN_API_KEY = 'replace-with-a-long-random-secret'
$env:DAN_API_LISTEN = '0.0.0.0:8080'
.\Start-DAN-Service.ps1
```

The default provider transport is the bundled encrypted peer network. To use a
Tailscale-encrypted direct activation ring instead, run:

```powershell
.\Start-DAN-Service.ps1 -ProviderNetwork tailscale
```

Point each provider at the coordinator's Tailscale address on port `50201` and
set `network=tailscale`. Providers advertise their ring listener on port `50202`;
the coordinator uses `50205` for the return leg.

CUDA graphs run inside compatible provider workers. Speculative decoding is not
enabled by the standard package: it requires a compatible local draft GGUF and a
coordinator build with llama.cpp support. Do not infer that it is active from
ring mode alone; `dan_speculative_enabled` reports the live setting.

The DAN API gateway supports authenticated `GET /v1/models` and
`POST /v1/chat/completions`, including `"stream": true` SSE responses. Terminate
TLS in a reverse proxy before exposing it to the internet.

Security
--------

Provider connections are encrypted and tied to their permanent PeerID. The coordinator
currently accepts any authenticated PeerID, then checks that the provider uses that same
identity inside DAN. A public relay is still required when the coordinator itself cannot
accept incoming connections through its router.

Release builders can put relay addresses in the package with
`release_windows_coordinator.ps1 -Relay RELAY_ADDRESS`. DAN reserves those
routes automatically and prints them with the direct addresses.

The bundled `runtime\dan-sidecar.exe` can operate that circuit-v2 relay on a
separate public Windows host. Run it with `-relay-service`, a persistent `-key`,
a public `-listen` port, and one repeated `-allow PEER_ID` for every coordinator
or provider permitted to reserve. See `P2P_TRANSPORT.md` in the source tree for
the command, limits, and open-relay warning.
