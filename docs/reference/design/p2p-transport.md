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
unused localhost ports, then starts the stage worker. The same sidecar accepts the
worker's local ring stream and carries it directly to the assigned successor over
the separate `/dan/ring/1.0.0` protocol. The receiver prepends the authenticated
libp2p PeerID; the assigned worker or coordinator rejects an unexpected predecessor.

Build the Windows download with the coordinator's direct and relay routes already
inside it:

```powershell
.\scripts\release_windows_provider.ps1 -CoordinatorPeer `
  '/dns4/coordinator.example/tcp/50200/p2p/COORDINATOR_PEER_ID,/dns4/relay.example/tcp/443/p2p/RELAY_PEER_ID/p2p-circuit/p2p/COORDINATOR_PEER_ID' `
  -Relay '/dns4/relay.example/tcp/443/p2p/RELAY_PEER_ID'
```

Add one or more `-Relay RELAY_ADDRESS` values when a relay is needed. The
provider reserves on those relays and advertises its circuit addresses for the
activation ring. Pass all known direct and relay addresses for the same
coordinator in `-CoordinatorPeer`, joined with commas; the helper tries them as
routes to the same checked PeerID. Packaging fails if a circuit coordinator route
has no relay reservation address.

## Operating a relay

`dan-sidecar` can also run the circuit-v2 relay that the coordinator and providers
reserve on. Put it on a host with a stable public TCP port and persistent identity.
Allowlist every coordinator/provider PeerID that may reserve; unlisted peers may
connect only to an allowed reservation and cannot consume their own relay slot:

```powershell
.\runtime\dan-sidecar.exe -key .\data\relay.key `
  -listen /ip4/0.0.0.0/tcp/50200 -relay-service `
  -allow COORDINATOR_PEER_ID -allow PROVIDER_1_PEER_ID -allow PROVIDER_2_PEER_ID
```

The default per-direction circuit limit is 1,024 MiB or 30 minutes. Change it with
`-relay-limit-mib` and `-relay-limit-duration` only after measuring real traffic.
`-allow-any` intentionally creates a bounded public relay; do not use it without
host-level monitoring, firewall limits, and an abuse policy. Publish the printed
public address as `/ip4/PUBLIC_IP/tcp/50200/p2p/RELAY_PEER_ID` (or its DNS form).

## Clean-machine acceptance

The remaining release proof needs two clean Windows hosts: run the coordinator
package and one provider on the first, and a second provider package behind a
different router/NAT on the other. Do not copy repository files or use build-tree
binaries.

Build the provider ZIP with `release_windows_provider.ps1 -CoordinatorPeer -Relay`
using the coordinator's stable direct and/or relay multiaddresses; a loopback
address is valid only with the lower-level package script for same-host tests.
The release builder rejects a loopback-only route list.

1. Keep each package script's generated `.zip.sha256` file and verify it before
   extraction (repeat for each archive):

   ```powershell
   $Zip = '.\DAN-Provider-v1.0.1-Windows-x64.zip'
   $Expected = ((Get-Content "$Zip.sha256" -Raw) -split '\s+')[0]
   if ((Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
       throw 'Package SHA-256 mismatch'
   }
   ```

   Extract both ZIPs, then run the provider's
   `dan-provider.exe --check --config .\config\provider.conf`.
2. Start `Start-DAN-Service.cmd`, then both providers. Record all three PeerIDs
   and whether each selected route was direct or relayed from the sidecar logs.
3. Require `/health` = `200`, one buffered completion, one SSE completion, and
   zero `dan_resident_sessions` and `dan_kv_memory_bytes` afterward.
4. Stop the remote provider during a request. Require `503`, restart the unchanged
   package and identity, then require `200`, a successful new request, and an
   increased `dan_replica_reformations_total`.
5. Attach coordinator/provider/sidecar logs, metrics before and after churn,
   archive hashes, Windows/GPU versions, elapsed recovery time, and any operator
   intervention. Any undocumented manual fix fails the clean-machine gate.

## Fallback

Leave out `--provider-peer-auth` and run the coordinator and workers with their
old TCP or Tailscale addresses. No DAN message changed.

## Current limit

In automatic served mode, a lost provider fails the active and queued requests,
then the coordinator accepts reconnecting or replacement providers and reforms the
replica without restarting. An idle metrics heartbeat detects loss without waiting
for a client request. Providers retain an unchanged loaded stage while
reconnecting, so an identical reassignment avoids another download and model load.
Persistent session state is not migrated across a reform yet; clients must create a
new session. Explicit-address and interactive modes still fail closed on provider
loss. Automatic ring mode uses either provider-advertised private endpoints or
libp2p multiaddresses; replacement providers rebuild the ring as part of replica
reformation. Cross-machine/NAT package acceptance is still pending.
