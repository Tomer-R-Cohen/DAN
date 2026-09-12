# DAN service operations

## Start and observe

Run `Start-DAN-Service.cmd`. Readiness is `GET /health`; it returns `200` only
when at least one configured replica is available. Authenticated `GET /metrics`
uses Prometheus text format. Configure the scraper with the same bearer token as
`DAN_API_KEY`, then load [`dan-alerts.yml`](../deploy/prometheus/dan-alerts.yml).
The Windows launcher prevents automatic system sleep while the service is running;
it does not prevent an operator-requested sleep, hibernation, shutdown, or power loss.
`dan_speculative_enabled`, `dan_pipeline_depth`, and the draft-token counters
show whether speculative decoding is actually active; ring readiness alone does not.

## Horizontal scale

Run one coordinator package and provider pool per independent replica group,
using the same model and strong API key. Keep each coordinator's binary port on
loopback, expose each DAN gateway only through TLS, and put a standard HTTP load
balancer in front of the gateways. Route traffic only to instances whose
`/health` returns `200`. This is the recommended multi-host layout because a
provider or coordinator failure removes only one replica group.

The gateway can alternatively balance several coordinators itself by repeating
`-coordinator`, but those binary endpoints have no application-layer
authentication and must be reachable only through an authenticated encrypted
private network:

```powershell
.\dan-api-gateway.exe -listen 127.0.0.1:8080 -model MODEL_ID `
  -coordinator 10.0.0.11:50100 -coordinator 10.0.0.12:50100
```

Do not expose coordinator port `50100` to the public internet. DAN chat requests
carry their full message history and are stateless at this gateway boundary;
coordinator-resident sessions are not shared across replica groups.

Release logs are under `data\logs` beside the coordinator package. Provider logs
are under `%LOCALAPPDATA%\DAN\logs`. Preserve the first error and the preceding
formation events when reporting an incident.

On an otherwise idle release deployment, run the gate for 24 hours while deliberately stopping and restarting
providers. A request interrupted by provider loss is counted as an outage and
retried after readiness returns. The gate fails if readiness does not recover
within three minutes, any unrelated request fails, or a stateless request leaves
resident session/KV allocation behind:

```powershell
$env:DAN_API_KEY = 'the-service-key'
.\Test-DAN-Soak.ps1 -ExpectedOutages 1
```

Use the default of zero when no deliberate churn is scheduled. The gate fails
if the observed outage count differs, so an unrelated transient outage cannot
be silently accepted.

## Alert actions

- `DANNoReplica`: inspect coordinator formation logs, then provider sidecar and
  worker logs. Confirm the advertised VRAM covers the active manifest.
- `DANCoordinatorDown`: restart only that coordinator group; other configured
  coordinators continue receiving requests.
- `DANQueueSaturated`: add an independent replica group and repeat its
  `-coordinator` address at the gateway. Raising the queue only delays rejection.
- `DANRequestFailures`: check provider disconnects, queue timeouts, and client
  cancellations separately before changing capacity.
- `DANReplicaChurn`: remove the unstable provider and verify its network path,
  GPU driver, power, and model cache before returning it to service.

## Backup, upgrade, and rollback

Back up `%LOCALAPPDATA%\DAN\coordinator\identity.key`; losing it changes the
coordinator PeerID and invalidates distributed provider packages. Back up
`%LOCALAPPDATA%\DAN\identity.key` and `provider-v1.1.conf` on each provider.
Model ranges are verified caches and can be downloaded again. Active sessions
and queued requests are memory-only and cannot be restored.

To upgrade, keep the old extracted directory, stop the service, extract the new
archive beside it, copy only `config\active-model.json` and `config\relays.txt`
when customized, then start the new service and wait for `/health` to return
`200`. Roll back by stopping it and starting the untouched old directory. The
identities and provider model caches live outside the release directories.
