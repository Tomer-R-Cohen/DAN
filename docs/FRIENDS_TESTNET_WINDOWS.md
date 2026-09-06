# DAN Provider for Windows

This package lets a Windows 10/11 x64 gaming PC with an NVIDIA GPU join Tomer's
trusted friends testnet. It is not for the public internet. DAN sets up its
private networking component; never forward the coordinator or RPC worker port
through your router.

## Set up once

1. Make sure the PC has a current NVIDIA driver.
2. Ask Tomer for the coordinator `HOST:PORT`.
3. Extract the complete ZIP to a normal folder. Do not move files out of it.
4. Double-click `dan-provider.exe`. DAN checks the GPU and installs its signed official private-network
   component if needed; approve Windows UAC and complete the one-time browser sign-in.
   Enter the coordinator address Tomer gives you. Later launches start directly.

The provider window uses normal-language states: Starting, Connecting, Available,
Preparing, Downloading, Loading, Contributing, Reconnecting, Action Required, or
Error. Available is healthy: it means DAN has no suitable work for the GPU yet.
Starting the provider twice is safe; the second copy reports that DAN is already
running. Temporary network loss and sleep/wake reconnect automatically with the
same provider identity and cache.

An internal PowerShell setup remains available for release diagnostics:

```powershell
powershell -ExecutionPolicy Bypass -File .\runtime\setup-provider.ps1
```

Setup discovers the PC's private IPv4 address, detects the GPU, and validates
the bundled worker. DAN itself stays non-administrator; Windows may elevate only
the signed network installer. DAN does not change the driver, firewall, router,
or antivirus.

Start later by double-clicking `dan-provider.exe`.
Press Ctrl+C or close its console window to stop contributing. DAN stops only
workers it started. Your provider identity, configuration, and downloaded model
cache remain for the next start.

## Storage and settings

DAN stores mutable data in `%LOCALAPPDATA%\DAN`:

```text
provider-id       stable random identity; do not delete it
provider.conf     settings
models\            verified model cache
logs\              diagnostic logs you choose to create
```

To reserve more GPU memory for Windows and games, stop DAN, edit
`reserve_vram_mib=1536` in `provider.conf`, and restart. The value must be less
than the GPU's real memory. DAN v1 does not detect games or adjust this while
running.

## Troubleshooting

- GPU detection fails: run `nvidia-smi`; update or repair the NVIDIA driver.
- Coordinator does not connect: confirm Tailscale is connected and recheck the
  private coordinator address with Tomer. DAN retries automatically.
- Worker fails: keep the complete `runtime` folder beside `dan-provider.exe`. Do not mix
  files from another llama.cpp build.
- Firewall: the coordinator must reach TCP port `50052` on this PC's Tailscale
  IPv4 address. Setup adds no rule. If Windows blocks it, ask Tomer to create one
  inbound TCP rule scoped to port 50052, the coordinator's Tailscale IPv4 source,
  and `rpc-server.exe`. Never open that port to every source or forward it on a router.
- Setup keeps old settings: edit `%LOCALAPPDATA%\DAN\provider.conf`, or rename it
  and rerun setup. Keep `provider-id` to retain the same identity.

Run a local check without contacting the coordinator:

```powershell
.\dan-provider.exe --check
```

Capture a diagnostic log:

```powershell
& .\dan-provider.exe *>&1 | Tee-Object "$env:LOCALAPPDATA\DAN\logs\provider.log"
nvidia-smi -q > "$env:LOCALAPPDATA\DAN\logs\nvidia-smi-q.txt"
```

Send Tomer those two files and the package version. Do not send Tailscale keys or
account credentials.

## Package compatibility

The package includes `dan-provider.exe` and a `runtime` directory containing the
internal managed provider, `rpc-server.exe`, required DLLs, and the official
Tailscale installer. The RPC worker comes from official llama.cpp release
`b10791`, which is DAN's pinned revision
`95ef7fc16054e63b427a3ef00188e055ef7586d8`; setup accepts an explicit prebuilt
worker path during development, but gamers should receive it already bundled.

Technical provider diagnostics are written to `%LOCALAPPDATA%\DAN\logs\provider.log`.
The package reports build `testnet-ui-v1`, protocol version 1, and retains the
pinned llama.cpp revision above. Logs never contain Tailscale authorization keys.
