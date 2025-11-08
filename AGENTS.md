### Orient
- Overview: read `README.md`; skim `core/*/*.py` and all `core/*/schema.py` to learn structure/contracts.
- Naming: files (and symbols) must be extensible, informative, self-explanatory.

### Structure
- Long-term scripts → `scripts/misc/`
- Short-term scripts → `scripts/legacy/<topic>/`
- Notes/reports → `data/scratchpad/`
- Temp data → `data/tmp/` (use a subfolder if many files)

### Develop
- Imports: add `import utils.path_utils` at the top of scripts.
- Run modules: `python -m A.B.C` from project root.
- Async from sync: `from utils.async_utils import run_coroutine; run_coroutine(func(*args, **kwargs))`.
- Finish: clean up; `ruff check --fix`; `black . --workers=1`.

### Strategy & Quality
- Plan: 3–6 steps; exactly one in progress.
- Reuse/extend/fix existing scripts/functions/classes; keep logic unified; avoid duplication.
- If unsure, stop and ask.
- Don’t catch exceptions that shouldn’t occur; let them surface.
- Validate with assert/sense-checks; if a validation isn’t supposed to fail, don’t handle—raise.
- NEVER use mock data or suppress unexpected errors.
- Prefer real, end-to-end examples; include key variable values.
- For analysis, include raw outputs (DataFrame `.describe()/.summary()/.head`, raw regression summaries).
- Run full-pipeline tests and inspect output with your own eyes. Iterate tirelessly until it all works.

### Communicate
- Keep updates short and action-focused; state success criteria for long ops.
- Final message: what changed/why, files touched, how to verify (exact command).

### Commands
- Lint: `ruff check --fix`
- Format: `black <specific_file> --workers=1` to format the file(s) that you changed. Never format files that you didn't change.
- Run: `python -m package.module --args`
