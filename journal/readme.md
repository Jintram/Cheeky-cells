


### 2026-06-09

The current version of this repo has rather bloated code, and needs some
refactoring.

Additionally, some conceptual choices could be reconsidered.


#### Refactoring

- Removing unused code
- Redesigning how test and training set are defined (without metadata xlsx)

#### Conceptual choices

- Rescaling per channel is probably not very detrimental, but unclear
which goal it precisely achieves. 
    - Currently baseline is subtracted per channel.
    - This doesn't add information.
    - Goal should be to equalize the data among all the different images.
    - Perhpas test performance with different settings??
    - Allow the rescaling function to be chosen by user. Supply a list of some good options.
- Previously, I used tiling to get more variety in training data, and avoiding
having to annotate large image files.
    - This is however baked into the core of the repo, and this feature
    is not used any more. 
    - Should be relocated to a pre-processing step.

#### Pruning unused code

*(Analysis largely by claude opus 4.7, executed manually.)*

Reachability analysis was done starting from the active
`pipelineclean_phase{1,2,3}_*.py` entry points. Everything below is
unreachable from those scripts (and not already parked under
`_previous_version/`, `old-code/`, or `deactivate/`). Convention: relocate via
`git mv` to `old-code/` rather than deleting, matching how legacy is already
handled in this repo.

Active entry points (in scope):
- `pipelineclean_phase1_example_arabtop.py`
- `pipelineclean_phase1_example_roots.py`
- `pipelineclean_phase2_example_arabtop.py`
- `pipelineclean_phase3_example_roots.py`
- `pipelineclean_phase3_example_roots_fortrain.py`

##### Phase A — move files to `old-code/`

Package code (zero importers in active code):

- [X] `git mv cheeky_cells/annotating_data/annotation_aided_OLD.py misc/old-code/refactor202606`
  — `_OLD` suffix; only the non-`_OLD` sibling is imported.
- [X] `git mv cheeky_cells/machine_learning/applying/apply_model.py misc/old-code/refactor202606`
  — only defines `resolve_device()`, never called. Then `git rm -r
  cheeky_cells/machine_learning/applying/` to drop the now-empty subpackage.
- [X] `git mv cheeky_cells/plotting/plotting_generic.py misc/old-code/refactor202606`
  — `colors_to_cmap()` + colour palettes; only referenced from
  `_previous_version/plotting_annotation.py`.

One-off / scratch scripts that don't belong in the package:

- Kept these files, but put them all in ./misc/ directory;
    - [X] `git mv cheeky_cells/devtools/showimage.py misc/old-code/refactor202606`; then
    `git rm -r cheeky_cells/devtools/` if `__init__.py` is all that remains.
    - [X] `git mv cheeky_cells/misc/example-arabidopsistop-mask.py misc/old-code/refactor202606`
    - [X] `git mv cheeky_cells/misc/napari_colormap_labellayer.py misc/old-code/refactor202606`
    - [X] `git mv cheeky_cells/misc/toying-background-corr-ara.py misc/old-code/refactor202606`
    — imports `cheeky_cells.*` internally but is itself never imported.
    - [X] `git mv cheeky_cells/misc/misc_functions.py misc/old-code/refactor202606`
    — `anonimize_filenames`, `relabeling_oldcode`, etc.; past-task utilities.
    Then `git rm -r cheeky_cells/misc/` if `__init__.py` is all that remains.

Legacy entry-point at repo root:

- [X] `git mv pipeline_example_arabidopsis_ML_phase2_train_model.py misc/old-code/refactor202606`
  — bypasses `orchestrate_phase2_clean.py` with its own `Phase2Config` and
  local `evaluate_on_tiles_and_plot`; superseded by
  `pipelineclean_phase2_example_arabtop.py`.

##### Phase B — update references

- [X] Edit [claude.md](../claude.md): remove
  `pipeline_example_arabidopsis_ML_phase2_train_model.py` from the
  phase-2 entry-point sentence; drop `cheeky_cells/devtools/` and/or
  `cheeky_cells/misc/` from the repository-layout bullet list if those
  subpackages were removed.
- [X] Check [cheeky_cells/__init__.py](../cheeky_cells/__init__.py) for any
  imports from removed subpackages (likely a no-op, but verify).

##### Phase C — verify

- [-] Do an import-only smoke run of each of the five active entry points
  (just up to the first orchestrator call + config construction — no need to
  open the napari GUI).
- [-] `grep -r "cheeky_cells.misc\|cheeky_cells.devtools\|cheeky_cells.machine_learning.applying\|plotting_generic\|annotation_aided_OLD" cheeky_cells/ pipelineclean_*.py`
  — must return zero hits.
- [X] `pip install -e .` in the `cheeky-all` conda env to surface any
  package-discovery breakage.

##### Out of scope for this pass

- `_previous_version/`, `old-code/`, `deactivate/` — already legacy.
- Pruning unused **symbols** inside reachable modules. Spot-checks confirmed
  e.g. `plot_overlay`/`plot_sidebyside`/`plot_dataset` inside
  `orchestrate_phase2_clean.py` ARE called from the same file. A symbol-level
  pass should be a separate task.
- The conceptual refactors above (rescaling, metadata xlsx, tiling).
- The `_tile_transform.npy` saving in `annotation_aided.py` — already tracked
  separately, decision pending.