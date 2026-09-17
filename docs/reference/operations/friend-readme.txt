DAN - share your GPU with the DAN network
==========================================

Start
  1. Unzip this folder anywhere.
  2. Double-click dan-provider.exe.
  3. If Windows asks about network access, click "Allow" (recommended).

That is all. No router settings, no port forwarding, no account.

What happens
  - DAN checks your NVIDIA GPU and keeps some memory free for you
    (reserve_vram_mib in config\provider.conf, default 1536 MiB).
  - It joins the DAN network through:
      {{BOOTSTRAP}}
  - The window shows a live dashboard: connection, what your GPU is doing,
    which DAN members it is linked to, speed and totals.
  - When someone runs a model, DAN downloads only the part your GPU runs and
    shows the progress on screen. Downloads are cached for next time.
  - Traffic goes directly to other members when possible, otherwise through the
    DAN relay.

Stop
  Press Ctrl+C or close the window. Your PC is freed immediately.

Requirements
  - Windows 10 or 11, 64-bit
  - An NVIDIA GPU with a current driver
  - Everyone must use the same DAN package version

Trouble?
  Logs: %LOCALAPPDATA%\DAN\logs  (send stage-worker.log and sidecar.log)
  "No supported NVIDIA GPU was found": update the NVIDIA driver.

Running models yourself (optional)
  Start-DAN-Client.ps1 -Bootstrap {{BOOTSTRAP}} `
      -Manifest config\provider-owned-qwen2.5-0.5b-q4km.json -- --chat
  Type a message and press Enter. /new starts over, /quit exits.
