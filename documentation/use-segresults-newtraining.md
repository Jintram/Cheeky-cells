


### Allow segmented data to be used as new training data

To achieve this, see example in `pipelineclean_phase3_example_roots_fortrain.py`
for adjustments to the phase 3 pipeline, which are

- `save_images=True` in config, such that additional output is generated that ensures generated annotation is compatible
train loop requirements.
- Make sure you have a convenient output directory, and run the script
as you would otherwise.

Adjustments to phase 1 are:
- Set `file_infix=''`, as original filenames should be recognized.
- Make sure you know where output files are put (typically, I renamed the subfolder
of phase 3 output, e.g. from `segfolder` to `segfolder_humancorr`) and adjust the link to the folder with segmentation files accordingly. See commented code in 
`pipelineclean_phase1_example_roots.py`, this can be exectued after setting 
up config.
```
# your config settings
config1 = o1.Phase1Config(
    ... 
)
# adjusted path
config1.segfolder = os.path.join(config1.outputdirectory, 'segfiles_humancorr/')
```

- Run the script as you would otherwise.

