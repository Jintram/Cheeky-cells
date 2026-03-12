
## Setup

I started a .venv folder using `python3 -m venv` and `python3 -m pip`, but 
I didn't finish this. Thus, the `pyproject.toml` file can be ignored.

Instead, I've used the Conda environment:
```bash
conda activate 2025_pytorch2
```




## Workflow description

### Annotating images

For this analysis to work, ground truth images need to be created to train the U-net. 

This requires manual annotation of images, but I've already observed good results with
using only 11 annoted images (which were subdivided over a training and test set), 
that were 2195x2195 pixels in size.

<img src=visuals/annotation_overview.png>
<b>Figure. </b><i>Overview of annotation procedure. <b>Left</b> the original image, <b>center</b> the human-generated annotation, and on the <b>right</b> training classes that were generated to train the model (this is optional and training can also be done directly on a human-annotated segmentation).</i>


## Random notes to incorporate later

- Napari shortcuts:
    - `[` and `]` chance brush size