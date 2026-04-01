# %% ################################################################################
# Arabidopsis ML pipeline - Phase 2
#
# Phase 2 goal:
# - train the segmentation model
# - save checkpoints and quality-control plots

from dataclasses import dataclass
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


# %% ################################################################################
# Phase 2 configuration container (isolated)

@dataclass
class Phase2Config:
    # Root directory of this repository
    script_dir: str = os.path.dirname(os.path.abspath(__file__))

    # Main output directory used by this phase
    outputdirectory: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/'

    # Input metadata used for train/test split of tiles
    metadata_customized_filename: str = 'metadata_imagefiles_manual20251022.xlsx'

    # Device and model setup
    target_device: str = 'mps'
    nr_classes: int = 5

    # Dataset settings
    img_suffix: str = '_tile_img_enhanced'
    lbl_suffix: str = '_tile_seg'
    artificial_n: int = 1000
    crop_size: int = 1000

    # Training settings
    learning_rate: float = 1e-3
    batch_size: int = 8
    epochs: int = 24
    lr_schedule_step_len: int = 8

    # Optional: load an existing checkpoint before training
    model_checkpoint_to_load: str | None = None

    # Plotting settings
    n_examples_to_plot: int = 10


# %% ################################################################################
# Phase 2 helpers


def get_output_paths(config: Phase2Config) -> dict:
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
    # No longer needed — cheeky_cells is installed as a package.
    pass


def resolve_device(requested_device: str) -> str:
    # Keep mps default but fall back to cpu when needed
    if requested_device == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def import_custom_modules_phase2():
    # Import only modules needed for phase 2
    from cheeky_cells.machine_learning.datasetclass import dataset_classes as cdc
    from cheeky_cells.machine_learning.model import unet_model as cunet
    from cheeky_cells.machine_learning.trainer import trainer as ct

    return cdc, cunet, ct


def get_plant_cmap() -> ListedColormap:
    # Class colors for prediction plots
    custom_colors_plantclasses = [
        '#000000',
        '#90EE90',
        '#FFFFFF',
        '#A52A2A',
        '#006400',
    ]
    return ListedColormap(custom_colors_plantclasses)


def sidebysideplot(current_img_rgb, current_prd, current_lbl, plot_name: str, pltfolder: str, cmap_plantclasses) -> None:
    # Save input / prediction / truth side-by-side figure
    cm_to_inch = 1 / 2.54
    fig, ax = plt.subplots(1, 3, figsize=(15 * cm_to_inch, 5 * cm_to_inch))
    ax[0].imshow(current_img_rgb)
    ax[1].imshow(current_prd[0].argmax(0), cmap=cmap_plantclasses, vmin=0, vmax=5)
    ax[2].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)
    for idx_ax in range(3):
        ax[idx_ax].set_xticks([])
        ax[idx_ax].set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(pltfolder, f'prediction_{plot_name}.pdf'), dpi=300)
    plt.close()


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


def load_training_metadata(config: Phase2Config) -> pd.DataFrame:
    # Load metadata used for train/test split
    metadata_filepath = os.path.join(config.outputdirectory, config.metadata_customized_filename)
    return pd.read_excel(metadata_filepath)


def build_train_test_datasets(df_metadata: pd.DataFrame, config: Phase2Config, output_paths: dict, cdc):
    # Build train dataset
    dataset_train = cdc.ImageDataset_tiles(
        df_metadata,
        output_paths['segfolder'],
        'train',
        img_suffix=config.img_suffix,
        lbl_suffix=config.lbl_suffix,
        transform=cdc.augmentation_pipeline_input,
        transform_label=cdc.augmentation_pipeline_label,
        targetdevice=config.target_device,
        ARTIFICIAL_N=config.artificial_n,
        CROP_SIZE=config.crop_size,
    )

    # Build test dataset
    dataset_test = cdc.ImageDataset_tiles(
        df_metadata,
        output_paths['segfolder'],
        'test',
        img_suffix=config.img_suffix,
        lbl_suffix=config.lbl_suffix,
        transform=cdc.augmentation_pipeline_input,
        transform_label=cdc.augmentation_pipeline_label,
        targetdevice=config.target_device,
        ARTIFICIAL_N=config.artificial_n,
        CROP_SIZE=config.crop_size,
    )

    return dataset_train, dataset_test


def initialize_unet_model(config: Phase2Config, cunet):
    # Initialize U-net model
    return cunet.UNet(n_channels=3, n_classes=config.nr_classes).to(config.target_device)


def custom_lr_schedule(epoch: int, step_len: int = 50):
    # Define piecewise LR scaling schedule
    lr_scalefactor = [1] * step_len + [0.1] * step_len + [0.01] * step_len
    if epoch >= step_len * 3:
        epoch = step_len * 3 - 1
    return lr_scalefactor[epoch]


def train_model(dataset_train, dataset_test, model_unet, config: Phase2Config, cdc, ct):
    # Build weighted loss from label distribution
    label_weights = cdc.get_label_weights(dataset_train, suffix_img=config.img_suffix, suffix_lbl=config.lbl_suffix)
    loss_fn = torch.nn.CrossEntropyLoss(label_weights)

    # Build optimizer
    optimizer = torch.optim.Adam(model_unet.parameters(), config.learning_rate)

    # Build data loaders
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True)

    # Build scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: custom_lr_schedule(epoch, step_len=config.lr_schedule_step_len),
    )

    # Run training and validation loops
    list_loss_tracker = []
    list_correct = []
    start_time_overall = time.time()

    for epoch_idx in range(config.epochs):
        print('=' * 30)
        print(f"Epoch {epoch_idx + 1}, LR: {scheduler.get_last_lr()}")
        start_time = time.time()

        loss_tracker = ct.train_loop(
            train_loader,
            model_unet,
            loss_fn,
            optimizer,
            len(dataset_train),
            config.batch_size,
        )
        current_correct = ct.test_loop(
            val_loader,
            model_unet,
            loss_fn,
            len(dataset_test),
            config.batch_size,
        )

        scheduler.step()
        list_loss_tracker.append(loss_tracker)
        list_correct.append(current_correct)

        elapsed_time = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - start_time))
        print(f"Epoch {epoch_idx + 1} completed in {elapsed_time}..")

    elapsed_time_overall = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - start_time_overall))
    print(f'Training completed in {elapsed_time_overall}..')
    print('Done!')

    return model_unet, list_loss_tracker, list_correct


def save_model_checkpoint(model_unet, output_paths: dict) -> str:
    # Save model with a timestamped filename
    current_time_formatted = time.strftime('%Y%m%d_%H%M')
    output_path = os.path.join(output_paths['modelfolder'], f'modelUNet{current_time_formatted}.pth')
    torch.save(model_unet.state_dict(), output_path)
    return output_path


def load_model_checkpoint(model_unet, checkpoint_path: str):
    # Load model weights from checkpoint file
    model_unet.load_state_dict(torch.load(checkpoint_path))
    return model_unet


def plot_training_history(list_loss_tracker, list_correct, epochs: int) -> None:
    # Plot batch-wise loss trace
    datay = np.array(list_loss_tracker).flatten()
    datax = np.array(range(len(datay))) * 100
    plt.plot(datax, datay, color='black')
    plt.ylim([0, np.max(datay) * 1.1])
    plt.axvline(datax[-1] / 3, linestyle='--', color='black')
    plt.axvline(datax[-1] / 3 * 2, linestyle='--', color='black')

    # Plot epoch-wise validation metric
    data_correctx = np.linspace(datax[-1] / epochs, datax[-1], epochs)
    data_correcty = np.array(list_correct)
    plt.plot(data_correctx, data_correcty)

    # Show figure
    plt.show()
    plt.close()


def evaluate_on_tiles_and_plot(model_unet, dataset_train, dataset_test, output_paths: dict, cmap_plantclasses, n_examples: int = 10):
    # Evaluate and save plots for train and test examples
    for whichone in ['test', 'train']:
        for idx in range(n_examples):
            if whichone == 'train':
                current_img, current_lbl = dataset_train[idx]
            else:
                current_img, current_lbl = dataset_test[idx]

            X = current_img[None, :, :, :]
            logits = model_unet(X)

            current_img = current_img.cpu().numpy()
            current_img_rgb = current_img.transpose(1, 2, 0)
            current_lbl = current_lbl.cpu().numpy()
            current_prd = logits.cpu().detach().numpy()

            correct_pixels = current_prd[0].argmax(0) == current_lbl
            accuracy = np.sum(correct_pixels) / correct_pixels.size
            print(f'Accuracy for {whichone} image {idx}: {accuracy * 100:.2f} %')

            plot_name = f'{whichone}_{idx:03d}'
            sidebysideplot(current_img_rgb, current_prd, current_lbl, plot_name, output_paths['pltfolder'], cmap_plantclasses)
            overlayplot(current_img_rgb, current_prd, current_lbl, plot_name, output_paths['pltfolder'], cmap_plantclasses)


# %% ################################################################################
# Phase 2 runner


def run_phase2_pipeline(config: Phase2Config) -> str:
    # Prepare path and outputs
    add_script_dir_to_path(config.script_dir)
    output_paths = get_output_paths(config)
    ensure_output_dirs(output_paths)

    # Resolve and print actual device
    config.target_device = resolve_device(config.target_device)
    print(f'Using target device: {config.target_device}')

    # Import custom modules
    cdc, cunet, ct = import_custom_modules_phase2()

    # Load metadata
    df_metadata = load_training_metadata(config)

    # Build datasets
    dataset_train, dataset_test = build_train_test_datasets(df_metadata, config, output_paths, cdc)

    # Initialize model
    model_unet = initialize_unet_model(config, cunet)

    # Optionally load checkpoint before further training
    if config.model_checkpoint_to_load is not None:
        model_unet = load_model_checkpoint(model_unet, config.model_checkpoint_to_load)

    # Train model
    model_unet, list_loss_tracker, list_correct = train_model(
        dataset_train,
        dataset_test,
        model_unet,
        config,
        cdc,
        ct,
    )

    # Save checkpoint
    saved_model_path = save_model_checkpoint(model_unet, output_paths)
    print(f'Saved checkpoint: {saved_model_path}')

    # Plot training history
    plot_training_history(list_loss_tracker, list_correct, config.epochs)

    # Plot prediction quality
    cmap_plantclasses = get_plant_cmap()
    evaluate_on_tiles_and_plot(
        model_unet,
        dataset_train,
        dataset_test,
        output_paths,
        cmap_plantclasses,
        n_examples=config.n_examples_to_plot,
    )

    return saved_model_path


# %% ################################################################################
# Execute phase 2 directly

if __name__ == '__main__':
    default_config = Phase2Config()
    run_phase2_pipeline(default_config)
