# MLB Model Preservation Design

## Goal

Preserve the fixed MLB analysis model so it can be reused after this chat is lost, reviewed by a human, validated by code, and published to GitHub without committing private or temporary data.

## Design

The repository will use `docs/model.md` as the single detailed model specification. A root `AGENTS.md` will tell future Codex sessions to load that document before analyzing games and not to modify the model unless the user explicitly asks.

A machine-readable `config/model.json` will store stable thresholds and source preferences that are useful to scripts. `analysis/pipeline.py` will validate that configuration and expose the fixed analysis-layer order plus a report skeleton. It will not scrape live sites or change the existing scoring behavior in `analysis/model.py`.

Human-facing templates will make daily output and postgame review consistent. Temporary HTML, virtual environments, cached Statcast data, and private reports will be excluded from Git.

## Components

- `AGENTS.md`: project-level instructions for future Codex sessions.
- `docs/model.md`: authoritative model rules.
- `config/model.json`: machine-readable stable defaults.
- `analysis/pipeline.py`: configuration loader, pitch-velocity grading helper, and analysis checklist builder.
- `docs/analysis-output-template.md`: default game-analysis structure.
- `docs/reviews/README.md`: postgame review format and model-change discipline.
- `docs/new-chat-prompt.md`: fallback prompt for environments that do not load `AGENTS.md`.
- `docs/data-sources.md`: source authority and cache policy.

## Testing

New pipeline behavior will be developed test-first in `tests/test_pipeline.py`. Existing `analysis/model.py` tests must continue to pass unchanged.

