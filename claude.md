# claude.md

Guidance for AI coding assistants (Claude Code, Copilot, etc.) working in this
repository. Keep entries short and factual; this file is loaded into the
agent's context on every turn, so brevity matters.

---

## Project overview

**Cheeky-cells** is a Python toolkit for training and applying a U-Net to
segment objects in microscopy / plant images. It is organized as a three-phase
pipeline:

- **Phase 1** — Manual / semi-automatic annotation of training data using Napari.
- **Phase 2** — Train a U-Net model on the annotated tiles.
- **Phase 3** — Apply the trained model to new images.

Each phase has an example entry-point script at the repo root
(`pipelineclean_phase1_example_*.py`, `pipeline_example_arabidopsis_ML_phase2_train_model.py`,
`pipelineclean_phase3_example_*.py`) and a matching orchestrator under
[cheeky_cells/orchestrators/](cheeky_cells/orchestrators/).

The package was originally developed for cell and plant segmentation but is
intended to be generic across segmentation problems.

---

## Repository layout

- [cheeky_cells/](cheeky_cells/) — installable Python package.
  - [orchestrators/](cheeky_cells/orchestrators/) — top-level pipeline drivers (`orchestrate_phase{1,2,3}_clean.py`).
  - [annotating_data/](cheeky_cells/annotating_data/) — Napari-based annotation, preliminary segmentation, post-processing.
  - [machine_learning/](cheeky_cells/machine_learning/) — model, dataset class, trainer, application code.
  - [prepostprocessing_input/](cheeky_cells/prepostprocessing_input/) — dataset-specific pre/post-processing (e.g. `ara_roots/`).
  - [plotting/](cheeky_cells/plotting/), [readwrite/](cheeky_cells/readwrite/), [misc/](cheeky_cells/misc/), [devtools/](cheeky_cells/devtools/) — supporting utilities.
- `pipelineclean_phase*_example_*.py` — runnable example pipelines at repo root. Users copy and edit these.
- [_previous_version/](_previous_version/) — legacy notebooks and modules. **Do not edit** unless explicitly asked; treat as reference only.
- [old-code/](old-code/) — superseded scripts. Same: reference only.
- [documentation/](documentation/) — design notes (`datasets.md`, `importing_functionality.md`, `use-segresults-newtraining.md`, `TODO.md`).
- [changelog/](changelog/), [journal/](journal/), [visuals/](visuals/) — bookkeeping folders.

Package metadata lives in [pyproject.toml](pyproject.toml). Dependencies are
**not** declared there — they are installed via conda (see below).

---

## Environment & commands

This project uses **conda**, not pip-managed virtualenvs, because of `pytorch`
and `napari`.

```bash
# E.g.
conda create -n cheeky-all -c conda-forge -c pytorch python=3.12 \
    numpy pandas matplotlib pytorch torchvision scipy scikit-image \
    napari pillow nd2 seaborn pyqt openpyxl -y

# Install this package in editable mode
conda activate cheeky-all
pip install -e .

# Run a pipeline
python pipelineclean_phase1_example_roots.py
```

There is also a local venv at `deactivate/` used in some terminals
(`source deactivate/bin/activate`). Prefer the conda environment for new work.

**No test suite, linter, or CI is configured.** Do not invent commands like
`pytest`, `ruff`, or `make test` — they will not exist. Validate changes by
running the relevant pipeline script or by importing the modified module.

---

## Conventions

- Each phase is driven by a `Config` dataclass (e.g. `Phase1Config`) constructed
  in the user's pipeline script and passed into orchestrator functions.
- Dataset-specific behavior (custom segmentation, custom Napari editor
  functions, custom colormaps) is plugged in by **assigning functions to config
  attributes** rather than subclassing. See
  [documentation/importing_functionality.md](documentation/importing_functionality.md).
- External project-specific code can be added to a `Config` by `pip install -e`
  of another repo and assigning its functions onto the config object.
- Image tiles and annotations are exchanged as `.npy` files; metadata as
  `.xlsx`. Filenames carry meaningful suffixes (e.g. `_tile.npy`,
  `_tile_transform.npy`, `_tile_humanseg.npy`).
- Module imports in pipeline scripts often include a commented
  `# import importlib; importlib.reload(...)` line — this is intentional for
  interactive (Spyder / Jupyter) development.

---

## Working preferences

- **Do not modify** files under `_previous_version/`, `old-code/`, or
  `deactivate/` unless explicitly asked.
- **Prefer editing existing files** over creating new ones; the package layout
  is already established.
- When adding dataset-specific code, place it under
  `cheeky_cells/prepostprocessing_input/<dataset>/` following the `ara_roots/`
  pattern.
- Do not add type hints, docstrings, or comments to existing code that does not
  already have them, unless that is the task.
- Do not introduce new dependencies without flagging it — the conda recipe in
  the README is the source of truth and must be updated in lockstep.

---

## Known gotchas

- `pyproject.toml` declares `dependencies = []`. Runtime deps come from conda;
  `pip install -e .` only registers the package.
- `file_formats` on configs must be a **tuple** — `('.tif',)` with a trailing
  comma for length 1.
- Napari opens a GUI window. Pipeline scripts that call `phase1_annotate` are
  **not** suitable for headless execution.
- The `_tile_transform.npy` files produced during annotation are currently
  unused downstream — see [/memories/repo/todo.md](/memories/repo/todo.md) before
  changing related code.

---

## Pointers for deeper context

- High-level usage and installation: [readme.md](readme.md)
- Plugging in external functionality: [documentation/importing_functionality.md](documentation/importing_functionality.md)
- Dataset format expectations: [documentation/datasets.md](documentation/datasets.md)
- Using prior model output as training input: [documentation/use-segresults-newtraining.md](documentation/use-segresults-newtraining.md)
- Open issues / design TODOs: [documentation/TODO.md](documentation/TODO.md)
