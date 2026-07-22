# AGENTS.md

## Cursor Cloud specific instructions

Truck Gate Scheduler is a standalone Python + Streamlit optimization app. There is no
database, backend API, or other external service — all state is local JSON under `data/`
and `config/`. The only long-running service is the Streamlit UI.

- Python venv lives at `.venv` (the update script recreates/refreshes it). Run every command
  from the repo root so `src` imports resolve.
- Run the app: `.venv/bin/streamlit run src/ui/app.py --server.headless true --server.port 8501`
  (headless flag avoids the interactive email prompt on first launch).
- Tests: `.venv/bin/pytest -q` (~40s; 51 tests). This is the full test/verify suite — there
  is no configured linter (no ruff/flake8/pylint config).
- Standard setup/run/API commands are documented in `README.md`; the dispatcher works out of
  the box using the committed `config/*.json` fallbacks, so no tuning/generation scripts are
  required before running or testing.
