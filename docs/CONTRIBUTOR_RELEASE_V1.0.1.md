DAN Provider v1.0.1 - Windows x64
================================

Requirements
------------

- Windows 10 or 11, 64-bit
- NVIDIA GPU with a current NVIDIA driver
- Enough free disk space for assigned model data
- Internet access

Start
-----

1. Extract the entire ZIP. Do not run the program from inside the ZIP.
2. Double-click `dan-provider.exe`.
3. Keep the DAN Provider window open while contributing.

There is no router setup, VPN installation, account, or coordinator address to
enter. The package already contains the coordinator's public identity and routes.
DAN creates this provider's identity on first start and reuses it on later starts.

The first model assignment may download several gigabytes. The dashboard shows
the assigned layers, progress, speed, ETA, cache state, VRAM, requests, tokens,
and network state. Valid cached model ranges are reused after restart.

Press Ctrl+C or close the window to stop. Provider identity, configuration, and
model cache are stored under `%LOCALAPPDATA%\DAN`.

Security
--------

The connection is encrypted and both ends have stable PeerIDs. A public pool
accepts new PeerIDs automatically; this proves who returned on later connections,
not whether their inference was correct. Only run packages from a source you trust.
