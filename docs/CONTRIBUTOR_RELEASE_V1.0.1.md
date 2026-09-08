DAN Provider v1.0.1 - Windows x64
================================

Requirements
------------

- Windows 10 or 11, 64-bit
- NVIDIA GPU with a current NVIDIA driver
- Enough free disk space for assigned model data
- A DAN Tailscale invitation and coordinator address

Start
-----

1. Extract the entire ZIP. Do not run the program from inside the ZIP.
2. Double-click `dan-provider.exe`.
3. Approve Tailscale installation/sign-in if requested.
4. Enter the coordinator's Tailscale address, including port 50200.
5. Keep the DAN Provider window open while contributing.

The first model assignment may download several gigabytes. The dashboard shows
the assigned layers, progress, speed, ETA, cache state, VRAM, requests, tokens,
and network state. Valid cached model ranges are reused after restart.

Press Ctrl+C or close the window to stop. Provider identity, configuration, and
model cache are stored under `%LOCALAPPDATA%\DAN`.

Security
--------

v1.0.1 is a trusted-tailnet testnet release. Do not expose coordinator port
50200 publicly. Only join a DAN network you trust.
