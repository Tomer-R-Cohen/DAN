DAN - share your GPU with the DAN network
==========================================

Start
  1. Run DAN-Setup.exe (no administrator rights needed).
  2. Open "DAN Node" from the desktop or the Start menu.
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
    shows the progress on screen. Downloads are kept for next time.
  - Traffic goes directly to other members when possible, otherwise through the
    DAN relay.

Chat
  Open "DAN Chat". It picks the largest model the GPUs online right now can run
  together. Type a message and press Enter. /new starts over, /quit exits.

Stop
  Press Ctrl+C or close the window. Your GPU is freed within a few seconds.

Requirements
  - Windows 10 or 11, 64-bit
  - An NVIDIA GPU, RTX 20-series or newer, with a current driver
  - Everyone must use the same DAN version

Trouble?
  Logs: %LOCALAPPDATA%\DAN\logs  (send stage-worker.log and sidecar.log;
  they can be copied while DAN is running)
  "No supported NVIDIA GPU was found": update the NVIDIA driver.
  Uninstall: "Uninstall DAN" in the Start menu, or Windows Settings > Apps.
