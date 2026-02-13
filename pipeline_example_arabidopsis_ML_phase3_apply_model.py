# %% ################################################################################
# Arabidopsis ML pipeline - Phase 3
#
# Phase 3 goal:
# - apply trained segmentation model to actual data
# - save overlay plots and predicted masks

from dataclasses import dataclass
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.colors import ListedColormap
from torchvision.transforms import ToTensor


# %% ################################################################################
# Phase 3 configuration container (isolated)

@dataclass
class Phase3Config:
    # Root directory of this repository
    script_dir: str = os.path.dirname(os.path.abspath(__file__))

    # Main output directory
    outputdirectory: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/'

    # Model settings
    target_device: str = 'mps'
    nr_classes: int = 5

    # Required checkpoint path for inference
    model_checkpoint_to_load: str | None = None

    # Input data metadata settings
    data_path_input: str = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/'
    regenerate_input_metadata: bool = True
    metadata_input_filepath: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/metadata_imagefiles_manual_inputdata20251027.xlsx'

    # Normalization + prediction settings
    bg_percentile: int = 10

    # One single-image demo prediction
    image_path_test_single: str = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/20250721batch1/20250721_OY_09.jpg'

    # Batch prediction settings
    n_random_input_files: int = 100


# %% ################################################################################
# Phase 3 helpers


def get_output_paths(config: Phase3Config) -> dict:
    # Build output folders used in this phase
    return {
        'segfolder': os.path.join(config.outputdirectory, 'humanseg/'),
        'modelfolder': os.path.join(config.outputdirectory, 'models/'),
        'pltfolder': os.path.join(config.outputdirectory, 'plots/'),
    }


def ensure_output_dirs(paths: dict) -> None:
    # Ensure output directories exist
    os.makedirs(paths['segfolder'], exist_ok=True)
    os.makedirs(paths['modelfolder'], exist_ok=True)
    os.makedirs(paths['pltfolder'], exist_ok=True)


def add_script_dir_to_path(script_dir: str) -> None:
    # Make repository modules importable
    if script_dir not in sys.path:
        sys.path.append(script_dir)


def resolve_device(requested_device: str) -> str:
    # Keep mps default but fall back to cpu when needed
    if requested_device == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def import_custom_modules_phase3():
    # Import only modules needed for phase 3
    import readwrite.cheeky_readwrite as crw
    import annotating_data.annotation_aided as caa
    import prepostprocessing_input.preprocessing_input as cpp
    from machine_learning.model import unet_model as cunet

    return crw, caa, cpp, cunet


def get_plant_cmap() -> ListedColormap:
    # Class colors for prediction overlays
    custom_colors_plantclasses = [
        '#000000',
        '#90EE90',
        '#FFFFFF',
        '#A52A2A',
        '#006400',
    ]
    return ListedColormap(custom_colors_plantclasses)


def overlayplot(current_img_rgb, current_prd, current_lbl, plot_name: str, pltfolder: str, cmap_plantclasses) -> None:
    # Save overlay figure with predicted labels over input
    if current_prd.ndim > 2:
        current_pred_lbl = current_prd[0].argmax(0)
    else:
        current_pred_lbl = current_prd

    current_pred_lbl_transl = np.pad(current_pred_lbl, ((0, 0), (20, 0)), 'constant', constant_values=0)
    img_shape_ratio = current_img_rgb.shape[1] / current_img_rgb.shape[0]

    cm_to_inch = 1 / 2.54
    if current_lbl is None:
        fig, ax = plt.subplots(1, 1, figsize=(10 * img_shape_ratio * cm_to_inch, 10 * cm_to_inch))
        ax.imshow(current_img_rgb)
        ax.imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0 * (current_pred_lbl_transl > 0))
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, ax = plt.subplots(1, 2, figsize=((5 * img_shape_ratio + 5) * cm_to_inch, 5 * cm_to_inch))
        ax[0].imshow(current_img_rgb)
        ax[0].imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0 * (current_pred_lbl_transl > 0))
        ax[1].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(pltfolder, f'predictionoverlay_{plot_name}.pdf'), dpi=300)
    plt.close()


def initialize_unet_model_for_inference(config: Phase3Config, cunet):
    # Initialize U-net for inference
    return cunet.UNet(n_channels=3, n_classes=config.nr_classes).to(config.target_device)


def load_model_checkpoint(model_unet, checkpoint_path: str):
    # Load model weights from checkpoint file
    model_unet.load_state_dict(torch.load(checkpoint_path))
    return model_unet


def load_input_metadata(config: Phase3Config, crw) -> pd.DataFrame:
    # Optionally regenerate metadata for input folder
    if config.regenerate_input_metadata:
        all_subfolders = glob.glob(pathname='*', root_dir=config.data_path_input)
        crw.gen_metadatafile(
            config.data_path_input,
            subdirs=all_subfolders,
            outputdirectory=config.outputdirectory,
            suffix='_inputdata',
            default_columns=['comments'],
        )

    # Load selected metadata file
    return pd.read_excel(config.metadata_input_filepath)


def read_input_img_file(df_metadata_input: pd.DataFrame, file_idx: int, crw, cpp, caa, bg_percentile: int):
    # Read image from metadata
    img_test = crw.loadimgfile_metadata(df_metadata_input, file_idx)

    # Crop to plate interior
    img_test_crop, _ = cpp.preprocess_getbbox_insideplate(img_test)

    # Normalize intensity
    img_test_crop_norm = caa.image_autorescale(img_test_crop, rescalelog=False, bg_percentile=bg_percentile)
    return img_test_crop_norm


def get_ml_prediction(img_input, the_model, target_device: str, showplot: bool, cmap_plantclasses):
    # Run inference and return predicted class labels
    with torch.no_grad():
        img_torch = ToTensor()(img_input).to(target_device)
        X = img_torch[None, :, :, :]
        logits = the_model(X)
        prd_full = logits.cpu().detach().numpy()
        prd_labels = prd_full[0].argmax(0)

    if showplot:
        plt.imshow(prd_labels, cmap=cmap_plantclasses, vmin=0, vmax=4)
        plt.show()

    return prd_labels


def apply_model_to_single_image(config: Phase3Config, model_unet, caa, cmap_plantclasses, output_paths: dict):
    # Load one test image and run inference
    from PIL import Image

    img_full = np.array(Image.open(config.image_path_test_single).convert('RGB'))
    img_full_norm = caa.image_autorescale(img_full, rescalelog=False, bg_percentile=config.bg_percentile)

    with torch.no_grad():
        img_full_torch = ToTensor()(img_full_norm).to(config.target_device)
        X = img_full_torch[None, :, :, :]
        logits = model_unet(X)
        prd_full = logits.cpu().detach().numpy()

    # Save overlay plot
    current_img = X[0].cpu().numpy()
    current_img_rgb = current_img.transpose(1, 2, 0)
    overlayplot(
        current_img_rgb,
        prd_full,
        None,
        plot_name='fullimage_test',
        pltfolder=output_paths['pltfolder'],
        cmap_plantclasses=cmap_plantclasses,
    )


def apply_model_to_random_input_batch(df_metadata_input: pd.DataFrame, config: Phase3Config, model_unet, crw, cpp, caa, cmap_plantclasses, output_paths: dict):
    # Build output folder for batch predictions
    output_prediction_folder = os.path.join(output_paths['pltfolder'], 'fullimages_predictions/')
    os.makedirs(output_prediction_folder, exist_ok=True)

    # Select random file indices
    random_file_idxs = np.random.choice(df_metadata_input.index, size=config.n_random_input_files, replace=False)

    # Loop over selected files
    for file_idx in random_file_idxs:
        print(f'Processing file idx {file_idx} ..')

        img_test_crop_norm = read_input_img_file(
            df_metadata_input,
            file_idx=file_idx,
            crw=crw,
            cpp=cpp,
            caa=caa,
            bg_percentile=config.bg_percentile,
        )

        img_pred_lbls = get_ml_prediction(
            img_test_crop_norm,
            the_model=model_unet,
            target_device=config.target_device,
            showplot=False,
            cmap_plantclasses=cmap_plantclasses,
        )

        # Save overlay plot
        print('Plotting prediction')
        overlayplot(
            img_test_crop_norm,
            img_pred_lbls,
            None,
            plot_name=f'fullimage_idx{file_idx:03d}',
            pltfolder=output_prediction_folder,
            cmap_plantclasses=cmap_plantclasses,
        )

        # Save raw predicted mask
        print('Saving prediction..')
        np.save(os.path.join(output_prediction_folder, f'predictedmask_idx{file_idx:03d}.npy'), img_pred_lbls)


# %% ################################################################################
# Phase 3 runner


def run_phase3_pipeline(config: Phase3Config) -> None:
    # Prepare path and outputs
    add_script_dir_to_path(config.script_dir)
    output_paths = get_output_paths(config)
    ensure_output_dirs(output_paths)

    # Resolve and print actual device
    config.target_device = resolve_device(config.target_device)
    print(f'Using target device: {config.target_device}')

    # Make sure checkpoint is provided
    if config.model_checkpoint_to_load is None:
        raise ValueError('Set config.model_checkpoint_to_load to a valid .pth checkpoint path before running phase 3.')

    # Import custom modules
    crw, caa, cpp, cunet = import_custom_modules_phase3()

    # Initialize model and load checkpoint
    model_unet = initialize_unet_model_for_inference(config, cunet)
    model_unet = load_model_checkpoint(model_unet, config.model_checkpoint_to_load)

    # Build colormap
    cmap_plantclasses = get_plant_cmap()

    # Run one single-image prediction
    apply_model_to_single_image(config, model_unet, caa, cmap_plantclasses, output_paths)

    # Run random-batch predictions
    df_metadata_input = load_input_metadata(config, crw)
    apply_model_to_random_input_batch(
        df_metadata_input,
        config,
        model_unet,
        crw,
        cpp,
        caa,
        cmap_plantclasses,
        output_paths,
    )


# %% ################################################################################
# Execute phase 3 directly

if __name__ == '__main__':
    default_config = Phase3Config()
    run_phase3_pipeline(default_config)
