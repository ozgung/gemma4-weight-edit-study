# Proposer sandbox (weight-edit study, round 2)

Each proposer runs as a separate headless Claude Code process (`claude -p`) under macOS `sandbox-exec`. The forward-pass
server (`scripts/weight_edit/wsp_server.py serve`) stays outside the sandbox and works unchanged: the proposer only
writes `requests/fwd_<k>.json` in its run folder and reads `requests/*.done.json` and `work/`.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/wsp_server.py init RUN_ID --proposer fable \
        --model claude-fable-5-1 --effort max --budget 25
    scripts/weight_edit/sandbox/launch_proposer.sh RUN_ID fable max path/to/brief.md [prompt.txt]

- **Packet (read-only):** `outputs/weight_edit/sandbox/RUN_ID/`, which is also the cwd. It holds `brief.md` (copied from
  BRIEF), `dossier.md`, `search/` (calls.jsonl, files/, MANIFEST.txt from `outputs/weight_edit/shared/search`),
  `wsp_kernel.py` (a copy with an absolute repo root, so it finds the model) and `scratch/` (writable).
  `shared/split.json` is not copied. The brief must use these packet paths, not `outputs/weight_edit/shared/...`
  (that path is denied).
- **Run folder (read/write):** `outputs/weight_edit/proposers/RUN_ID/`. The launcher writes `transcript.jsonl`
  (stream-json, the full record for the audit) and `launcher.log` there.
- **Readable:** the packet, the run folder, the model directory, the MLX venv and its uv Python, Claude Code's binary and
  config, the keychain, the run's temp dir. **Denied:** the rest of the home directory, including the whole repo,
  other proposer runs, `~/.claude/projects` (other sessions' transcripts and memories), `history.jsonl`,
  file-history, paste-cache, sessions, session-env, shell-snapshots, jobs, todos and daemon; all of `/private/tmp`
  (other sessions' scratchpads and task outputs, `cc-socks`) except the run's temp dir; `/Users/Shared`, `/Volumes`.
- **Writes:** only the run folder, `scratch/`, the run's temp dir, and Claude Code's own state (`~/.claude` minus
  `projects/` other than this cwd's slug, minus `settings.json`).
- **Claude Code flags:** env cleared (`env -i`, so no `CLAUDE_CODE_MESSAGING_*`); `CLAUDE_CODE_TMPDIR` set to the run's
  temp dir; `--setting-sources ""` (no user or project settings, so no hooks, permissions or plugins);
  `--strict-mcp-config` with no servers (no claude.ai connectors); `--tools "Bash Read Write Edit Glob Grep"` (no
  Agent, WebFetch, WebSearch or messaging); `--permission-mode bypassPermissions` (the OS sandbox is the boundary).
  Unix-socket connections to `/tmp/cc-socks` and `~/.claude` are denied.
- **Waiting on passes:** a background until-loop works in `-p` mode. The process stays alive and the model resumes when
  the loop ends (tested with a 75 s wait).

Known holes: `file-read-metadata` is allowed everywhere, so `stat` of a guessed path reveals existence, size and mtime
(directory listings are denied). `~/.claude.json`, `~/.claude/settings.json`, plugins and skills are readable: config,
not study data. Outbound TCP is open (needed for the API), so the proposer could reach the internet through Bash
(`curl`). The keychain is readable at the file level (needed for OAuth); the `security` tool still enforces item ACLs.
