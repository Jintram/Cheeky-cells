#### Metadata-handling cleanup

##### TL;DR

Slimmed `df_metadata` from a grab-bag of mostly-unused columns to four:
`subdir`, `filename`, `segmentation_channel`, `train_or_test`. Collapsed
the two near-identical scanner helpers into a single recursive
`gen_metadatafile`. Phase 3 no longer round-trips (dumps in xlsx and then loads 
it again) its metadata through
xlsx — it's now built into the config. Dataset-wide constants
(`basedir`, `invert_image`) moved off the per-row table and onto the
phase config / loader arguments. 

Net effect: less duplication, fewer
flags, but a few breaking signature changes for anyone with their own
scripts on the old API (see end of file).

##### Context

Follow-up to the dead-code prune in
[refactoring_202606_A.md](./refactoring_202606_A.md). Focus: simplify the
metadata table (`df_metadata`) and the helpers that produce / consume it in
[cheeky_cells/readwrite/cheeky_readwrite.py](../cheeky_cells/readwrite/cheeky_readwrite.py),
plus the orchestrators and annotation modules that depend on them.

Motivation: the schema had drifted into a grab-bag (`basedir`, `subdir`,
`filename`, `segmentation_channel`, `invert_image`, plus `default_columns`
sprinkled by phase-1 only). Several columns were either dataset-wide
constants duplicated per row, or unused downstream. Two near-identical
generators (`gen_metadatafile` for phase 1, `gen_metadatafile_segfiles` for
phase 3) had also accumulated.

##### Step 1 — Strip mandatory schema in `gen_metadatafile`

- Dropped `default_columns` parameter and the `dataset` / `comments`
  columns it added by default. They were never read by any orchestrator;
  users who want extra columns can add them by editing the generated xlsx
  by hand.
- Dropped `invert_image` column. Inversion is dataset-wide (decided once
  per phase config), not per-file, so it now lives on the config and is
  passed into `loadimgfile_metadata(..., do_invert=...)`.
- Dropped `suffix` parameter; replaced with explicit `output_filename` so
  callers control the full filename rather than a magic infix.
- Final schema: `subdir`, `filename`, `segmentation_channel`,
  `train_or_test`.

##### Step 2 — Phase-3 metadata stays in memory

Phase 3 was round-tripping its auto-generated metadata through an xlsx file
even though nothing manual ever happened in between (no human editing
step, unlike phase 1).

- `Phase3Config.metadata_input_filepath: str | None` →
  `df_metadata: pd.DataFrame | None`.
- `collect_filelist` now populates `config.df_metadata` directly from
  `gen_metadatafile(..., save_xlsx=False)`.
- Removed `load_input_metadata` from
  [orchestrate_phase3_clean.py](../cheeky_cells/orchestrators/orchestrate_phase3_clean.py).
- `segment_all_files` strips `df_metadata` from the yaml dump dict (a
  DataFrame doesn't serialise cleanly anyway, and the per-row info is
  rederivable from the input directory).

##### Step 3 — Collapse the two generators

`gen_metadatafile_segfiles` was identical in spirit to `gen_metadatafile`;
the only real difference was a recursive vs. one-level glob.

- Removed `gen_metadatafile_segfiles`.
- Phase 3 now calls the unified `gen_metadatafile`.

##### Step 4 — Remove the redundant `basedir` column

The `basedir` column was the same value (`config.inputdirectory` /
`config.data_path_input`) on every row — a config constant masquerading as
per-row data.

- Dropped `basedir` from the schema.
- Added a required `basedirectory` argument to `loadimgfile_metadata`.
- Threaded `basedirectory` through the annotation entry points
  ([annotation_aided.py](../cheeky_cells/annotating_data/annotation_aided.py)):
  `annotate_pictures_aided` and `annotate_all_pictures_aided` both gained
  a required `basedirectory` parameter, and the phase-1 orchestrator
  passes `config1.inputdirectory` into them.
- `get_fileinfo_metadata` return tuple shrank from 4 to 3 elements:
  `(subdir, filename, segchannel)`. Callers updated:
  `loadimgfile_metadata`, `loadsegfile_metadata`, `savesegfile_default`,
  `annotate_pictures_aided`.

##### Step 5 — `train_or_test` is unconditional

`gen_metadatafile` had an `add_split_column=False` flag that phase 1 set
to `True` and phase 3 didn't bother with.

- Dropped the flag; `train_or_test` is always present (empty string by
  default), in the DataFrame and in the saved xlsx.

##### Step 6 — Recursive scan is the only mode

The two old generators differed only in whether they recursed. Step 3
preserved both modes behind a `recursive` flag; step 6 drops the flag.

- Removed `subdirs` and `recursive` parameters from `gen_metadatafile`.
- Function now always recurses under `basedirectory` and stores per-file
  paths relative to it in the `subdir` column.
- Phase-1 caller no longer passes `subdirs=None`; phase-3 caller no
  longer passes `recursive=True`.

> Heads-up: phase 1 will now also pick up images sitting in subfolders of
> `inputdirectory`, not just the top level. Previously it only globbed one
> level deep. Existing phase-1 input dirs that happen to have stray
> images in subfolders will start showing those rows in the metadata
> table.

##### Validation

No test suite, so validation was:

- `get_errors` (pylance) on every modified file — only pre-existing
  warnings remain (unrelated `is not` / semicolon style nits, one
  unused-import warning in `orchestrate_phase3_clean.py`).
- Smoke test under the `cheeky-all` conda env: import all three
  orchestrators + `cheeky_readwrite` + both annotation modules; inspect
  signatures of `gen_metadatafile`, `get_fileinfo_metadata`,
  `loadimgfile_metadata`, `collect_filelist`,
  `annotate_all_pictures_aided`; check that the removed names
  (`gen_metadatafile_segfiles`, `load_input_metadata`) are gone.

##### Files touched

- [cheeky_cells/readwrite/cheeky_readwrite.py](../cheeky_cells/readwrite/cheeky_readwrite.py)
  — `gen_metadatafile` rewrite, `gen_metadatafile_segfiles` removed,
  `get_fileinfo_metadata` returns 3-tuple, `loadimgfile_metadata` gains
  `basedirectory` (required) and `do_invert`, `loadsegfile_metadata` and
  `savesegfile_default` updated to 3-tuple destructure.
- [cheeky_cells/orchestrators/orchestrate_phase1_clean.py](../cheeky_cells/orchestrators/orchestrate_phase1_clean.py)
  — `Phase1Config` lost `invert_image` and `default_columns`;
  `phase1_setup` and `phase1_annotate` updated to new signatures and pass
  `basedirectory=config1.inputdirectory` into the annotation call.
- [cheeky_cells/orchestrators/orchestrate_phase3_clean.py](../cheeky_cells/orchestrators/orchestrate_phase3_clean.py)
  — `Phase3Config.metadata_input_filepath` → `df_metadata`,
  `load_input_metadata` removed, `collect_filelist` rewritten,
  `get_input_img_file` calls `loadimgfile_metadata` with
  `basedirectory=config.data_path_input`, `segment_all_files` strips
  `df_metadata` from the yaml dump.
- [cheeky_cells/annotating_data/annotation_aided.py](../cheeky_cells/annotating_data/annotation_aided.py)
  — `annotate_pictures_aided` and `annotate_all_pictures_aided` both
  gained a required `basedirectory` parameter; internal
  `get_fileinfo_metadata` destructure updated to 3-tuple.

##### Breaking-change call-out for downstream code

Anyone with their own scripts hitting the old API will need to update:

- `gen_metadatafile`: drop `default_columns`, `invert_image`, `suffix`,
  `subdirs`, `recursive`, `add_split_column` kwargs. Use `output_filename`
  for filename control and `save_xlsx=False` to get a DataFrame without
  writing to disk.
- `get_fileinfo_metadata`: now returns `(subdir, filename, segchannel)` —
  3 elements, not 4. The `basedir` element is gone.
- `loadimgfile_metadata`: `basedirectory` is now a required positional
  argument.
- `annotate_pictures_aided` / `annotate_all_pictures_aided`:
  `basedirectory` is now a required argument.
- `gen_metadatafile_segfiles` no longer exists — call `gen_metadatafile`
  instead.

##### Out of scope

- The pre-existing pylance warnings noted above.
- Symbol-level pruning inside reachable modules (still tracked from
  refactor A).
