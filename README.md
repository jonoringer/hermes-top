# hermes-top

`hermes-top` is a small, top-style CLI for watching Hermes Agent activity from Hermes' own persisted state instead of the host OS process table.

It reads Hermes session metadata from `~/.hermes/state.db` and infers active Hermes-owned work by looking for:

- assistant tool calls that have been issued but do not yet have a matching tool result
- long-running background terminal/process jobs surfaced in Hermes tool results
- open Hermes sessions with no current in-flight tool call, if you opt into idle rows

## Why this approach

This monitor stays Hermes-scoped:

- it does not scan `ps`
- it only lists work that Hermes itself has recorded
- it works on Linux, macOS, and other environments where Hermes uses the same SQLite state store

Because it is reading Hermes state rather than sampling kernel metrics, the `resource_hint` column is a heuristic:

- `network` usually means a web/API call
- `cpu-likely` usually means a shell or local compute task
- `gpu-likely` is inferred from command/tool text like `cuda`, `torch`, `vllm`, or similar markers

If you later want exact CPU/GPU telemetry, Hermes itself will need to emit that into its own state or expose a dedicated runtime endpoint. This first version is designed to be a safe Hermes-only monitor.

## Installation

```bash
pipx install git+https://github.com/jonoringer/hermes-top.git
```

`pipx` gives you an isolated CLI install without needing to manage a virtual environment manually.

## Usage

Run the live table:

```bash
hermes-top
```

One-shot snapshot:

```bash
hermes-top --once
```

If there are no active operations but Hermes has open sessions, the empty state tells you how many idle sessions were found and suggests `--include-idle`.

JSON output:

```bash
hermes-top --json
```

Include idle open sessions:

```bash
hermes-top --include-idle
```

Use a non-default Hermes database:

```bash
hermes-top --db-path /path/to/state.db
```

## Development

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Running on the Hermes machine

If Hermes uses the default state location, run:

```bash
hermes-top
```

If Hermes stores state somewhere else, point directly at it:

```bash
hermes-top --db-path /path/to/.hermes/state.db
```

Useful checks on the GPU box:

```bash
ls -la ~/.hermes
find ~ -path "*/.hermes/state.db" 2>/dev/null
hermes-top --once
hermes-top --include-idle --once
hermes-top --json
```

## Output columns

- `STATUS`: running, idle, or the background job status Hermes reported
- `KIND`: `web`, `tool`, `process`, or `session`
- `DURATION`: elapsed time since Hermes recorded the start of the call/job/session
- `SOURCE`: Hermes source like `cli`, `cron`, `discord`, etc.
- `TOOL`: tool name or command label when Hermes exposed one
- `RESOURCE`: Hermes-derived hint such as `network`, `cpu-likely`, or `gpu-likely`
- `SESSION`: Hermes session title
- `DETAIL`: compact argument or job detail summary

## Next step for deeper integration

If you want `hermes-top` to become exact instead of heuristic, the best upstream change in Hermes would be:

1. Emit a dedicated runtime activity table or JSON feed for in-flight tool calls.
2. Persist background process lifecycle updates with stable job IDs.
3. Optionally attach self-reported CPU/GPU counters for Hermes-managed workers.

This CLI is already structured so it can swap from inference to first-class Hermes telemetry later.
