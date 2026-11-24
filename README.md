# local-model-radar

The local counterpart of [model-radar](https://github.com/casareanderson/model-radar). On a hosted API, "free" means zero price. On an Ollama box with a 6 GB GPU, "free" means **fits in VRAM** — a 27B model that offloads to CPU is a slowdown, not a saving.

Two deterministic checks, same philosophy: gather facts, never ask a model its opinion.

- **`local-model-radar`** lists candidate models, filters them by hardware fit, pulls + probes the new ones (a real `get_weather` tool call, tokens/second, context length), and optionally adopts the best through a gate.
- **`local-model-canary`** watches the live local model — is Ollama up, is the model resident, does it still serve tool calls — and fails over through a local ladder when the GPU dies or the model is evicted.

## Why it exists

The home Ollama host is a physical pop-os box with an **RTX 2060 (6 GB VRAM)**, 31 GB RAM, i7-10875H. What has actually bitten this estate:

- **GPU driver mismatch** (kernel module vs NVML) silently dropped every model to CPU — "unreachable" looked like "alive" because `systemctl` said `active`.
- **Model eviction**: Hermes' `qwen3.5` and Honcho's `mistral-nemo` fight over 6 GB, so a turn that lands after the other consumer pays a full reload.
- **A model that answers but never tool-calls** is useless to an agent, however fast it is.

So the radar treats "fits in VRAM" as the hard budget, and the canary treats a transient as noise and only acts on N consecutive hard failures.

## Layout

```
local-model-radar/
├── bin/
│   ├── local-model-radar     # entry point -> localradar.radar
│   └── local-model-canary    # entry point -> localradar.canary
├── localradar/
│   ├── radar.py              # discovery, hardware gate, adoption
│   ├── canary.py             # Ollama up? model serving? ladder failover
│   ├── probe.py              # tool-call probe, tok/s, context, fit classifier
│   ├── discover.py           # Ollama library + config candidates
│   ├── config.py             # defaults + TOML merge
│   ├── notify.py             # stdout + Discord webhook
│   └── adapters/
│       └── hermes.py         # hermes-agent profile writer (local/custom)
└── config.example.toml
```

## Quick start

Prereqs: Python 3.11+ (`tomllib`), `PyYAML` (for the `hermes` adapter). No API key — it talks to your own Ollama over HTTP.

```bash
git clone <this repo> && cd local-model-radar
cp config.example.toml config.toml
./bin/local-model-radar --dry-run --include qwen3.5:9b llama3.2:3b
```

`--dry-run` pulls nothing it hasn't already got, writes no state, and reports each candidate's fit (`vram` / `partial`), tool-call result, tok/s and context.

Then try the canary (it needs a local primary to watch, which is any Hermes profile on `custom`):

```bash
./bin/local-model-canary --dry-run
```

Run them on timers — the canary hourly, the radar weekly:

```cron
0  * * * *  cd /opt/local-model-radar && ./bin/local-model-canary  >> /var/log/lm-canary.log 2>&1
30 4 * * 1  cd /opt/local-model-radar && ./bin/local-model-radar  >> /var/log/lm-radar.log 2>&1
```

## Configuration

Settings load from built-in defaults, then `--config` / `$LOCALRADAR_CONFIG` / `./config.toml`. Secrets are never read from the TOML file.

| Key | Default | What it does |
|---|---|---|
| `ollama.base_url` | `http://192.168.8.152:11434` | Local Ollama endpoint. |
| `hardware.vram_budget_mib` | `5500` | A model must fit under this to count as "in VRAM". |
| `radar.adopt` | `false` | Allow adoption at all. |
| `radar.adopt_targets` | `[]` | Profiles that may be repointed. Never `global`/CoS. |
| `radar.adopt_tokens_per_sec_floor` | `8.0` | Minimum tok/s to adopt. |
| `radar.adopt_speed_factor` | `0.7` | A candidate must be at least this fraction of the incumbent's speed. |
| `radar.blocklist` | `[]` | Models to skip (known CPU-only ones). |
| `canary.targets` | `[]` | Empty = every profile the adapter sees. |
| `canary.fails_before_failover` | `2` | Consecutive hard failures before failing over. |
| `canary.ladder` | `[]` | Ordered smaller models to fall back through. |
| `canary.local.model` | `""` | Final local fallback. |
| `notify.discord_webhook_env` | `LOCALRADAR_DISCORD_WEBHOOK` | Env var holding the webhook URL. |

## The three rules (carried over from model-radar)

1. **A 200 is not a capability test.** The probe asks for one specific tool call and checks the name and argument.
2. **Unreachable is not dead.** A busy/timeout model is left unseen and retried; a transient canary result never counts toward failover.
3. **Free is not a saving if it is slow.** Tok/s is measured against the incumbent in the same run.

## Status and limits

- Discovery is best-effort: `ollama.com/library` has no machine-readable catalogue, so the radar scrapes names and relies on `/api/show` for size/context **after** a pull. Use `candidates`/`--include` for the models you actually care about.
- Adoption never targets `global` or the CoS — those are a human call, same as model-radar.
- `max_probes_per_run` caps how many models get pulled per run (a pull can be GBs).
- No automated tests in this repo.

## Licence

MIT. Uses [Ollama](https://ollama.com) over HTTP and targets [hermes-agent](https://hermes-agent.nousresearch.com) profiles via PyYAML.
