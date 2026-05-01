# %% ################################################################################
# Arabidopsis ML pipeline - Phase 2
#
# Phase 2 goal:
# - train the segmentation model
# - save checkpoints and quality-control plots

import cheeky_cells  

from dataclasses import dataclass
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# custom modules
from cheeky_cells.machine_learning.datasetclass import dataset_classes as cdc
from cheeky_cells.machine_learning.model import unet_model as cunet
from cheeky_cells.machine_learning.trainer import trainer as ct
    # import importlib; importlib.reload(ct)


# %% ################################################################################
# Phase 2 configuration container (isolated)

def naparicmap_to_mplcmap(cmap_custom, nr_classes=None):
    """ Convert Napari cmap color dict to matplotlib listedcolormap. """
    # cmap_custom=config2.cmap_custom; nr_classes = config2.nr_classes
    
    # Convert transparent/none to black
    NAPARI_TO_MPL = {'transparent': (0, 0, 0, 1), 'none': (0, 0, 0, 1)}
    
    # Convert (sorted to ensure correct label order)
    colors = [NAPARI_TO_MPL.get(cmap_custom[k], cmap_custom[k]) for k in sorted(cmap_custom)]
    
    # match nr. of colors with nr. of classes, if nr. classes given
    max_lvls = np.min([len(colors), nr_classes])
    
    return ListedColormap(colors[:(max_lvls)])

@dataclass
class Phase2Config:

    # Main output directory used by this phase
    outputdirectory: str 

    # Input metadata used for train/test split of tiles
    metadata_customized_filename: str 

    # Device and model setup    
    nr_classes: int
    target_device: str = 'mps'

    # Output (sub)folders
    segfolder: str = None # defaults to 'humanseg/' during setup
    modelfolder: str = None # defaults to 'models/' during setup 
    pltfolder: str = None # defaults to 'plots_training/' during setup

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
    cmap_custom: dict = None # for napari
    cmap_custom_mpl: ListedColormap = None # for matplotlib
    
    def __post_init__(self):
        # (__post_init__ is automatically run after the dataclass is initialized, 
        # leaving original __init__ in place.)
            
        # Set up subdirs for input/output    
        self.segfolder = os.path.join(self.outputdirectory, 'humanseg/')
        self.modelfolder = os.path.join(self.outputdirectory, 'models/')
        self.pltfolder = os.path.join(self.outputdirectory, 'plots_training/')
        os.makedirs(self.segfolder, exist_ok=True)
        os.makedirs(self.modelfolder, exist_ok=True)
        os.makedirs(self.pltfolder, exist_ok=True)

        # Convert Napari cmap to matplotlib cmap
        self.cmap_custom_mpl = naparicmap_to_mplcmap(self.cmap_custom, self.nr_classes)


# %% ################################################################################
# Phase 2 plotting related functions


def plot_sidebyside(current_img_rgb, current_prd, current_lbl, plot_name: str, pltfolder: str, cmap_plantclasses) -> None:
    # pltfolder =  config2.pltfolder;  cmap_plantclasses = config2.cmap_custom_mpl
    
    # Save input / prediction / truth side-by-side figure
    cm_to_inch = 1 / 2.54
    fig, ax = plt.subplots(1, 3, figsize=(15 * cm_to_inch, 5 * cm_to_inch))
    ax[0].imshow(current_img_rgb)
    ax[1].imshow(current_prd[0].argmax(0), cmap=cmap_plantclasses) #, vmin=0, vmax=5)
    ax[2].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=np.max(current_lbl))
    plt.imshow(current_lbl, cmap=cmap_plantclasses)#, vmin=0, vmax=np.max(current_lbl))
    for idx_ax in range(3):
        ax[idx_ax].set_xticks([])
        ax[idx_ax].set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(pltfolder, f'prediction_{plot_name}.pdf'), dpi=300)
    plt.close()


def plot_overlay(current_img_rgb, current_prd, current_lbl, plot_name: str, pltfolder: str, cmap_plantclasses) -> None:
    
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


def plot_training_history(config2, list_loss_tracker, list_correct, list_test_loss_tracker=None):
    
    # Plot batch-wise loss trace
    datay = np.array(list_loss_tracker).flatten()
    datax = np.array(range(len(datay))) * 100
    plt.plot(datax, datay, color='black', label='Train loss')
    plt.ylim([0, np.max(datay) * 1.1])
    plt.axvline(datax[-1] / 3, linestyle='--', color='black')
    plt.axvline(datax[-1] / 3 * 2, linestyle='--', color='black')

    # Plot epoch-wise validation metric
    # percentage correct
    data_correctx = np.linspace(datax[-1] / config2.epochs, datax[-1], config2.epochs)
    data_correcty = np.array(list_correct)
    plt.plot(data_correctx, data_correcty, color='lightblue', label='test %correct')
    # loss function
    if list_test_loss_tracker is not None:
        plt.plot(data_correctx, list_test_loss_tracker, linestyle=':', color='black', label='test loss')

    plt.xlabel('Batch')
    plt.ylabel('Loss / Accuracy')
    plt.ylim([0,1])
    # legend
    plt.legend(loc='upper right')

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(config2.pltfolder, f'traininghistory.pdf'), dpi=300)
    plt.close()

def plot_overlay_contour(config2, image, label):
    
    fig, axs = plt.subplots(1,2, figsize=(10/2.54,5/2.54))
    _ = axs[0].imshow(image)
    _ = axs[1].imshow(image)
    #_ = axs[1].imshow(label, alpha=(label>0)*.5)
    _ = axs[1].contour(label, levels=(np.max(label)),
                    cmap=config2.cmap_custom_mpl,
                    linewidths=.15)
    
    # Now save this to the plot folder
    plt.tight_layout()
    
    return fig

def plot_overlay_contour2(config2, image, pred, label):
    """Plots orginal image and prediction on top, truth to the side"""
    
    fig, axs = plt.subplots(1,2, figsize=(10/2.54,5/2.54))
    
    _ = axs[0].set_title("Prediction")
    _ = axs[0].imshow(image)
    #_ = axs[1].imshow(label, alpha=(label>0)*.5)
    _ = axs[0].contour(pred, levels=(np.max(pred)),
                    cmap=config2.cmap_custom_mpl,
                    linewidths=.3)

    _ = axs[1].set_title("Truth")
    _ = axs[1].imshow(image)
    #_ = axs[1].imshow(label, alpha=(label>0)*.5)
    _ = axs[1].contour(label, levels=(np.max(label)),
                    cmap=config2.cmap_custom_mpl,
                    linewidths=.3)
    
    # Now save this to the plot folder
    plt.tight_layout()
    
    return fig

def plot_dataset(config2, dataset, prefix=""):
    # dataset = dataset_train; prefix="train"
    # dataset = dataset_test; prefix="test"
    
    for file_idx in range(len(dataset.filelist_imgs)):
        # file_idx = 0
        
        filename_img = dataset.filelist_imgs[file_idx]
        filename_lbl = dataset.filelist_labels[file_idx]
        
        # sample_idx, file_name = 0, dataset.filelist_imgs[0]
        image   = np.load(dataset.datadir + filename_img, allow_pickle=True)
        label   = np.load(dataset.datadir + filename_lbl, allow_pickle=True)

        fig = plot_overlay_contour(config2, image, label)

        plt.savefig(os.path.join(config2.pltfolder, f'plot-{prefix}_{filename_img}.pdf'), dpi=600)        
        plt.close(fig)
                       



################################################################################
# %% Helpers

def build_train_test_datasets(df_metadata: pd.DataFrame, config2: Phase2Config):
    
    # Build train dataset
    dataset_train = cdc.ImageDataset_tiles(
        df_metadata,
        config2.segfolder,
        'train',
        img_suffix=config2.img_suffix,
        lbl_suffix=config2.lbl_suffix,
        transform=cdc.augmentation_pipeline_input,
        transform_label=cdc.augmentation_pipeline_label,
        targetdevice=config2.target_device,
        ARTIFICIAL_N=config2.artificial_n,
        CROP_SIZE=config2.crop_size,
    )

    # Build test dataset
    dataset_test = cdc.ImageDataset_tiles(
        df_metadata,
        config2.segfolder,
        'test',
        img_suffix=config2.img_suffix,
        lbl_suffix=config2.lbl_suffix,
        transform=cdc.augmentation_pipeline_input,
        transform_label=cdc.augmentation_pipeline_label,
        targetdevice=config2.target_device,
        ARTIFICIAL_N=config2.artificial_n,
        CROP_SIZE=config2.crop_size,
    )

    return dataset_train, dataset_test


def initialize_unet_model(config2: Phase2Config, cunet):
    # Initialize U-net model
    return cunet.UNet(n_channels=3, n_classes=config2.nr_classes).to(config2.target_device)


def custom_lr_schedule(epoch: int, step_len: int = 50):
    # Define piecewise LR scaling schedule
    lr_scalefactor = [1] * step_len + [0.1] * step_len + [0.01] * step_len
    if epoch >= step_len * 3:
        epoch = step_len * 3 - 1
    return lr_scalefactor[epoch]


def train_model(config2, dataset_train, dataset_test, model_unet):
    
    # Build weighted loss from label distribution
    label_weights = cdc.get_label_weights(dataset_train, suffix_img=config2.img_suffix, suffix_lbl=config2.lbl_suffix)
    loss_fn = torch.nn.CrossEntropyLoss(label_weights)

    # Build optimizer
    optimizer = torch.optim.Adam(model_unet.parameters(), config2.learning_rate)

    # Build data loaders
    train_loader = DataLoader(dataset_train, batch_size=config2.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=config2.batch_size, shuffle=True)

    # Build scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: custom_lr_schedule(epoch, step_len=config2.lr_schedule_step_len),
    )

    # Run training and validation loops
    list_loss_tracker = []
    list_correct = []; list_test_loss_tracker = []
    start_time_overall = time.time()

    for epoch_idx in range(config2.epochs):
        print('=' * 30)
        print(f"Epoch {epoch_idx + 1}, LR: {scheduler.get_last_lr()}")
        start_time = time.time()

        loss_tracker = ct.train_loop(
            train_loader,
            model_unet,
            loss_fn,
            optimizer,
            len(dataset_train),
            config2.batch_size,
        )
        current_correct, current_test_loss = ct.test_loop(
            val_loader,
            model_unet,
            loss_fn,
            len(dataset_test),
            config2.batch_size,
        )

        scheduler.step()
        list_loss_tracker.append(loss_tracker)
        list_correct.append(current_correct)
        list_test_loss_tracker.append(current_test_loss)

        elapsed_time = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - start_time))
        print(f"Epoch {epoch_idx + 1} completed in {elapsed_time}..")

    elapsed_time_overall = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - start_time_overall))
    print(f'Training completed in {elapsed_time_overall}..')
    print('Done!')

    return model_unet, list_loss_tracker, list_correct, list_test_loss_tracker


def save_model_checkpoint(model_unet, output_paths: dict) -> str:
    # Save model with a timestamped filename
    current_time_formatted = time.strftime('%Y%m%d_%H%M')
    output_path = os.path.join(output_paths['modelfolder'], f'modelUNet{current_time_formatted}.pth')
    torch.save(model_unet.state_dict(), output_path)
    return output_path




def evaluate_on_tiles_and_plot(config2, model_unet, dataset_train, dataset_test, n_examples: int = 10):
    
    # Evaluate and save plots for train and test examples
    for whichone in ['test', 'train']:
        for idx in range(n_examples):
            # whichone='test'; idx=0
            
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
            plot_sidebyside(current_img_rgb, current_prd, current_lbl, plot_name, config2.pltfolder, config2.cmap_custom_mpl)
            plot_overlay(current_img_rgb, current_prd, current_lbl, plot_name, config2.pltfolder, config2.cmap_custom_mpl)
            
            # Overlay plot 2
            fig = plot_overlay_contour2(config2, current_img_rgb, current_prd[0].argmax(0), current_lbl)
            plt.savefig(os.path.join(config2.pltfolder, f'predictionoverlay2_{plot_name}.pdf'), dpi=600)
            plt.close(fig)


# %% ################################################################################
# Phase 2 runners


def phase2_setup(config2: Phase2Config) -> str:

    # Load metadata
    df_metadata = pd.read_excel(os.path.join(config2.outputdirectory, config2.metadata_customized_filename))

    # Build datasets
    dataset_train, dataset_test = build_train_test_datasets(df_metadata, config2)

    # Initialize model
    model_unet = cunet.UNet(n_channels=3, n_classes=config2.nr_classes).to(config2.target_device)

    # Optionally load checkpoint before further training
    if config2.model_checkpoint_to_load is not None:
        model_unet.load_state_dict(torch.load(config2.model_checkpoint_to_load))
        
    return dataset_train, dataset_test, model_unet

def phase2_train(config2, dataset_train, dataset_test, model_unet):

    # Train model
    model_unet, list_loss_tracker, list_correct, list_test_loss_tracker = \
        train_model(
            config2,
            dataset_train,
            dataset_test,
            model_unet        
        )

    # Save checkpoint
    saved_model_path = save_model_checkpoint(model_unet, output_paths)
    print(f'Saved checkpoint: {saved_model_path}')

    # Plot training history
    plot_training_history(config2, list_loss_tracker, list_correct, list_test_loss_tracker)

    # Plot prediction quality
    evaluate_on_tiles_and_plot(
        config2, 
        model_unet,
        dataset_train,
        dataset_test,
        n_examples=config2.n_examples_to_plot,
    )

    return saved_model_path


# %% ################################################################################
# Execute phase 2 directly

if __name__ == '__main__':
    default_config2 = Phase2Config()
    run_phase2_pipeline(default_config2)
