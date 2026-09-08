DAN Coordinator v1.0.1 - Windows x64
====================================

Requirements
------------

- Windows 10 or 11, 64-bit
- Tailscale connected to the same trusted tailnet as the providers
- At least one available DAN provider with enough contributed VRAM

Start
-----

1. Extract the entire ZIP.
2. Double-click `dan-coordinator.exe`.
3. Allow private-network access if Windows Firewall asks.
4. Give contributors your Tailscale IPv4 address with port 50200.
5. Wait while DAN selects providers and they download their assigned model ranges.
6. When `REPLICA READY` appears, type directly into the coordinator window.

Commands: `/new` resets the conversation, `/stats` shows current totals, and
`/quit` stops the coordinator. Providers reconnect automatically afterward.

Model selection
---------------

The default is Qwen2.5 0.5B Q4_K_M. Other tested Qwen2 model configurations are
in `config\models`. To switch, stop the coordinator and copy the selected JSON
over `config\active-model.json`, then start it again. The coordinator downloads
metadata only; providers download their assigned weight ranges.

If one provider can fit the selected model, DAN runs it there. Larger models are
automatically divided across additional providers.

Security
--------

v1.0.1 is a trusted-tailnet testnet release. Do not expose port 50200 publicly.
