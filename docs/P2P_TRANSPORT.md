# Peer-to-peer transport

`dan-sidecar` replaces the public TCP hop, not DAN's inference code. DAN still owns
the model, stage plan, messages, sessions, KV cache, and results. The sidecar sees
only a provider's checked identity and an opaque byte stream.

## Build

Go 1.25.7 and go-libp2p 0.48.0 are pinned in `sidecar/go.mod`.

Windows:

```powershell
.\scripts\build_sidecar.ps1
```

Linux:

```bash
bash ./scripts/build_sidecar.sh
```

Both scripts test the sidecar and build Windows and Linux AMD64 files.

## Run

First create each identity and record the printed PeerID:

```powershell
dan-sidecar.exe -key identity.key -id
```

Start the coordinator on localhost with identity checking enabled:

```text
dan-provider-owned-coordinator ... --provider-listen 127.0.0.1:50201 --provider-peer-auth
```

Put the coordinator sidecar in front of it. Repeat `-allow` for every provider:

```text
dan-sidecar -key coordinator.key -listen /ip4/0.0.0.0/tcp/50200 -inbound 127.0.0.1:50201 -allow PROVIDER_PEER_ID
```

For a public contribution pool, replace the `-allow` values with `-allow-any`.
Every connection is still encrypted and identified by its PeerID; new identities
do not require manual enrollment.

On each provider, forward a local port to the coordinator. A DNS name avoids a
manually managed IP address:

```text
dan-sidecar -key provider.key -listen /ip4/0.0.0.0/tcp/0 -forward 127.0.0.1:50200=/dns4/coordinator.example/tcp/50200/p2p/COORDINATOR_PEER_ID
dan-stage-worker --coordinator 127.0.0.1:50200 ...
```

The packaged `dan-provider` performs both commands itself. Its bundled
`config/provider.conf` contains `network=libp2p` and the coordinator addresses.
On first start it creates `%LOCALAPPDATA%\DAN\identity.key`, starts the sidecar on
an unused localhost port, then starts the unchanged stage worker against that port.

Build the Windows download with the coordinator's direct and relay routes already
inside it:

```powershell
.\scripts\release_windows_provider.ps1 -CoordinatorPeer `
  '/dns4/coordinator.example/tcp/50200/p2p/COORDINATOR_PEER_ID,/dns4/relay.example/tcp/443/p2p/RELAY_PEER_ID/p2p-circuit/p2p/COORDINATOR_PEER_ID'
```

Add one or more `-relay RELAY_ADDRESS` values when a relay is needed. Pass all
known direct and relay addresses for the same coordinator after `-forward`, joined
with commas. The helper tries them as routes to the same checked PeerID.

## Fallback

Leave out `--provider-peer-auth` and run the coordinator and workers with their
old TCP or Tailscale addresses. No DAN message changed.

## Current limit

A broken connection can be opened again, but the current coordinator stops
accepting providers after it forms a replica. A connection lost during inference
still ends that coordinator run. Fixing that belongs in the C++ lifecycle, not in
the networking helper.
