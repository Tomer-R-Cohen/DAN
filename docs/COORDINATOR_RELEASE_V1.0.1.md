DAN Coordinator v1.0.1 - Windows x64
====================================

Requirements
------------

- Windows 10 or 11, 64-bit
- At least one available DAN provider with enough contributed VRAM

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

If one provider can fit the selected model, DAN runs it there. Larger models are
automatically divided across additional providers.

Security
--------

Provider connections are encrypted and tied to their permanent PeerID. The coordinator
currently accepts any authenticated PeerID, then checks that the provider uses that same
identity inside DAN. A public relay is still required when the coordinator itself cannot
accept incoming connections through its router.

Release builders can put relay addresses in the package with
`release_windows_coordinator.ps1 -Relay RELAY_ADDRESS`. DAN reserves those
routes automatically and prints them with the direct addresses.
