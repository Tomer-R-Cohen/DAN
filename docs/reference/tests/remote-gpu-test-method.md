# Remote cross-machine testing: what to test next, and how this session did it fast

Date: 2026-09-13/14

This documents two things from the same session: a punch list of provider-owned
runtime behavior still worth testing, and the SSH-driving method that made a
real Windows-RTX-2070-plus-rented-RunPod-GPU test fast to iterate on instead of
requiring two people in two terminals.

## What was proven this session

- Genuine two-GPU split inference forced by real VRAM limits (no artificial
  caps) on Qwen2.5-14B-Instruct-Q8_0 (14.63 GiB) across a Windows RTX 2070 and
  a rented RunPod RTX A4000, over the open internet, encrypted and
  peer-authenticated via `dan-sidecar`, no VPN.
- CUDA graph reuse (confirmed from worker logs, not just assumed from docs).
- Speculative decoding + pipelining: 4.70 -> 12.11 tok/s (2.58x) on the same
  cross-GPU link, 46.7% draft-token acceptance, same cached weights reused
  for both runs (no re-download).
- Persistent range-backed weight cache survives a coordinator restart.
- NAT relay (circuit-v2): a Windows peer behind home NAT reserved a relay slot
  on the RunPod-side sidecar and obtained a dialable circuit address.

## What is NOT yet proven and is worth testing next

In rough priority order:

1. **The automatic activation ring** (`--ring-listen`/`--ring-proxy`/
   `--ring-target`/`--ring-return`), which bypasses the coordinator on the
   token-generation hot path. Got as far as a live protocol-framing bug in the
   tail-to-coordinator return leg (`bad frame header` -- the peer-ID header
   parses correctly but the frame immediately after it does not). Needs live
   byte-level instrumentation (temporary logging or a packet capture) to
   pin down, not more source reading. See "one real code gap found" below for
   the specific fix already applied and verified to build.
2. **Chunked prefill** (`--prefill-chunk N`) -- gated behind ring mode working
   at all (`docs` says "ring mode only"), so it's blocked by (1).
3. **Stranger-machine acceptance** -- this session's pod was operator-launched
   and operator-SSH'd; nobody ran a packaged release ZIP on a genuinely clean
   host. `docs/DAN_LAUNCH_READINESS.md` still lists this open.
4. **Bounded concurrent API / SSE / cancellation** over a real WAN link --
   this session only used one-shot `--prompt --requests 1` CLI batch mode, not
   `--listen` server mode with concurrent clients.
5. **Automatic recovery/reformation under real churn** -- kill a provider
   mid-request on the real cross-machine ring and confirm the replacement
   reforms and a subsequent request still succeeds. Only same-machine churn
   is in the existing evidence base.
6. **dan-api-gateway horizontal routing** over this same cross-machine setup
   (round-robin/failover across independent coordinator groups) -- not
   exercised; this session talked to the coordinator directly.
7. **The 32B model over this exact link**, to see whether the WAN network
   cost (145 ms/token baseline on 14B) dominates even harder at 32B, and
   whether speculative decoding's relative win holds, shrinks, or grows.
8. **Multiple concurrent resident sessions / persistent multi-turn** across
   the real cross-GPU link (`--resident-sessions N`, `--persistent`).

## One real code gap found and fixed

The coordinator's `--ring-return` is a single string used both to bind its own
listen socket and as the literal address handed to the tail provider to dial
-- there was no way to make those differ, which breaks cross-machine ring use
without Tailscale (a Tailscale IP happens to be dialable and bindable as the
same string; a public-IP-mapped-to-different-internal-port setup, like
RunPod's, is not). Fixed by exposing the coordinator's already-existing
`ring_target` field as a `--ring-target` flag (mirrors the worker's own
existing `--ring-target` flag), one line in `src/provider_owned/coordinator.cpp`:

```cpp
else if (option == "--ring-return") options.ring_return = value;
else if (option == "--ring-target") options.ring_target = value;   // added
```

This lets the coordinator advertise its own already-public sidecar address to
the tail instead of a literal loopback string. Verified it builds cleanly and
does not change behavior when the flag is omitted. The ring's remaining
framing bug (above) is a separate, deeper issue this change did not fix.

## Other real bugs found along the way (useful for the next test)

- `git -C <dir> apply <relative-patch>` resolves the patch path relative to
  `<dir>`, not the caller's cwd -- fixed in
  `docs/PROVIDER_OWNED_V2_WINDOWS_LINUX_TEST.md` to use `"$PWD/patches/..."`.
- `pkill -f <pattern>` matches its own invoking shell's command line when that
  exact pattern text appears in the command you sent over SSH (because the
  remote `sh -c "..."` process's own argv contains it) -- this killed the SSH
  session instead of the target process, repeatedly, until switched to
  killing by exact PID. Never `pkill -f` a name that also appears in the
  command line doing the killing.
- Under `--provider-peer-auth`, `--provider-id` must exactly equal the
  worker's own libp2p PeerID, or the coordinator silently logs
  "rejected invalid or duplicate provider" with no hint why.
- `--provider-listen` mode requires `--metadata-cache FILE` even though the
  usage string's bracket placement makes it look optional next to
  `--provider-peer-auth`.
- The VRAM planner's reserve is `max(1024 MiB, 15% of offered)` -- any
  `--vram-mib` below 1024 makes a provider mathematically unable to hold even
  one layer, and it just sits registered forever with no error explaining why.
- Mixing `--provider-peer-auth` off with a sidecar-forwarded (PeerID-prefixed)
  connection crashes the coordinator with an opaque `bad frame header`
  instead of a clear message about the mismatch.
- A one-shot coordinator (`--requests 1`) races ahead as soon as *any*
  sufficient replica forms -- if you're trying to force a multi-provider
  split and one provider alone has enough VRAM, it'll register and complete
  solo before a slower second provider even joins, silently defeating the
  test. Cap VRAM (above the 1024 MiB floor) or use a model too large for any
  single provider to force a real split.

## The SSH-driving method

The reason this ran fast (many iterations in one sitting, versus a human
manually alt-tabbing between two machines per docs like
`PROVIDER_OWNED_V2_WINDOWS_LINUX_TEST.md`) was driving both machines from one
place instead of two people/terminals:

1. **Non-interactive SSH from the Windows machine's own shell** reaches the
   rented pod directly: `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
   "<command>"`. No separate terminal, no copy-pasting between windows -- one
   agent issuing commands to both sides in the same conversation.
2. **Every long-running process on the pod was started as
   `nohup <cmd> > out.log 2>&1 < /dev/null & disown; echo LAUNCHED`**, then
   *immediately* backgrounded at the calling end too
   (`ssh ... "..." & sleep 1; ssh ... "cat out.log"`). Without `nohup`+
   `disown`+ full stdio redirection, the process dies the moment that SSH
   session closes, or the SSH command itself never returns because the
   channel stays open waiting for a child that still holds its stdio.
3. **Checking status is a fresh, separate SSH call** (`cat out.log`,
   `ps -eo pid,args | grep ...`) rather than trying to keep one long-lived SSH
   session open and interactively poll it. Cheap, stateless, and immune to
   any single command hanging.
4. **Kill by exact PID, captured from `ps`, never by `pkill -f <name>`**
   when that name might appear in the very SSH command doing the killing (see
   bug list above).
5. **A background-monitor loop** (poll a log for a completion/failure marker
   every N seconds, one final notification when it appears) instead of
   manually re-checking -- used for both the 14 GiB download waits and the
   ring-handshake retry loop, so the agent kept working on other prep (or the
   user could keep talking) while a multi-minute operation ran.
6. **The two machines' state was cross-referenced constantly**: a PeerID
   printed on the pod fed directly into the next command run on Windows, and
   vice versa, without the user needing to manually relay values back and
   forth -- both sides were visible in the same place at the same time.

None of this needed the pod's own web terminal or a second physical monitor.
The practical requirements were: the pod's SSH keys already trusted (added to
RunPod's account settings once), and knowing to background/detach every
remote process explicitly rather than relying on an interactive shell staying
open.
