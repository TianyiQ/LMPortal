## Project Overview
- Please read `README.md` for the project overview. Read all the `core/*/*.py` files to understand the full project. Read all the `core/*/schema.py` files to understand the general structure of the project.
- Always name your files in a way that is extensible, informative, and self-explanatory.

## Project Structure
- Scripts of long-term value should be placed in `scripts/misc/`. 
- Scripts for short-term use should be placed in subdirectories under `scripts/legacy/`.
- Notes and reports should be placed in `data/scratchpad/`.
- Temporary data should be placed in `data/tmp/` (if large in count, create a new subdirectory under `data/tmp/`).

## Strategy Tips
- Instead of creating new scripts every time, try to reuse, extend, or fix existing scripts if possible. The same goes for methods, functions, classes - try to reuse, extend, or fix existing ones if possible.
- If you are not 100% sure about whether you are doing the right thing the right way, stop, admit you can't proceed with confidence, and ask the user for help.
- If there isn't supposed to be an exception thrown at a certain point, don't try to catch it. We need to see the error to fix it.
- Perform extensive validations/sense-check assertions in your code. If there isn't supposed to be a failure of validation, don't try to do error handling; just throw an exception.
- NEVER use mock data or otherwise try to suppress unexpected errors.
- If possible, show a concrete example of the whole process of how your script runs on an actual data sample, including the values of various variables and strings. Avoid mock data.
- When doing analysis, try to include as much raw data as possible, including e.g. the dataframe `.describe()` / `.summary()` / `.head()` output, the raw regression summary of statsmodels, etc.
- Keep the logic unified whenever possible! Avoid duplicating code.
- Run full-pipeline tests and inspect output with your own eyes. Iterate tirelessly until it all works.

## Development Conventions
- Do `import utils.path_utils` at the beginning of your script to fix importation path issues. Always use `python -m A.B.C` to run your script so that the working directory is the project root.
- Run an coroutine function from non-async context by using `from utils.async_utils import run_coroutine; run_coroutine(func(*args, **kwargs))`.
- After you finish your work, always remember to:
  1. Clean up;
  2. Run `ruff check --fix` to fix any linting errors, and manually fix any remaining errors.
  3. Run `black <specific_file> --workers=1` to format the file(s) that you changed. Never format files that you didn't change.