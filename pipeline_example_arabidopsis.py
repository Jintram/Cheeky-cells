

# %% ################################################################################
# Some settings

import sys
import os

SCRIPT_DIR = '/Users/m.wehrens/Documents/git_repos/_UVA/2025_Cheeky-cells/'

# Directory with image files
basedirectory = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/'
subdirs = ['SELECTION_ML/Cropped/']

# Directory to place output
outputdirectory = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/'
segfolder = outputdirectory + 'humanseg/'
modelfolder = outputdirectory + 'models/'
pltfolder = outputdirectory + 'plots/'
os.makedirs(modelfolder, exist_ok=True)
os.makedirs(segfolder, exist_ok=True)
os.makedirs(pltfolder, exist_ok=True)

# add scripts in this folder to path
import sys
if SCRIPT_DIR not in sys.path: sys.path.append(SCRIPT_DIR)

# %% ################################################################################

# standard libs
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import glob

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
from torch.optim.lr_scheduler import StepLR, LambdaLR


# custom libs
import readwrite.cheeky_readwrite as crw
    # reload
    # import importlib; importlib.reload(crw)

import annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
    
import annotating_data.annotation_postprocessing as cap
    # import importlib; importlib.reload(cap)
    
cm_to_inch = 1/2.54

# %% ################################################################################

if False:
    # First generate the metadata file
    crw.gen_metadatafile(basedirectory, subdirs, outputdirectory)
        # This will auto-generate a metadata file with all image files in the specified folder.
        # It should be manually edited, and stored as an excel file (at metadata_filepath)

# Now update the metadata file location, after editing the auto-generated file and saving under custom name
metadata_customized_filename = 'metadata_imagefiles_manual20251022.xlsx'
metadata_filepath            = outputdirectory + metadata_customized_filename

# Now load the (manually updated) metadata
df_metadata = pd.read_excel(metadata_filepath)

# %% ################################################################################

# import importlib; importlib.reload(caa)

import annotating_data.dedicated_segmentation as cds

# Loop over the files to create the ground truth 
#
# Current classifciation
# 0 = background
# 1 = shoot
# 2 = root
# 3 = seed
# 4 = leaf
#
# This also creates the rescaled images
caa.annotate_all_pictures_aided(df_metadata, 
                                output_segfolder=outputdirectory + 'humanseg/',
                                intitial_segfolder=None,
                                TILE_SIZE=1000,
                                tile_selection_by='maxvar',
                                segfn=cds.basicplantseg1,
                                ignore_saved_file=False,
                                showplots=False,
                                rescalelog=False, bg_percentile=10, showrawimg=True)
# output_segfolder=outputdirectory + 'humanseg/'; intitial_segfolder=None; TILE_SIZE=1000; tile_selection_by='maxvar';
# segfn=caa.basicseg2; ignore_saved_file=False; showplots=False; rescalelog=False; bg_percentile=10





# %% ################################################################################
# Now let's post-process the segmented files to get an improved training mask

# A FUNCTION COULD BE ADDED HERE TO IMPROVE THE SEGMENTATIONS AUTOMATICALLY
# FOR THE ARABIDOPSIS DATA, THIS WAS NOT DONE..

# Commented out because not necessary (postprocess_basicsegfile_all() is aimed at binary masks of cells)
# Improve all basic segmentations, ie add boundaries proximity zones
# cap.postprocess_basicsegfile_all(df_metadata, segfolder, suffix='_seg_postpr', showplot=True)

# %% ################################################################################
# Now set up a dataset, model, and start training

### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
### CODE BELOW IS STILL MESSY XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# %% ######################################################################

# !!! TO DO!!! # !!! TO DO!!! # !!! TO DO!!! # !!! TO DO!!! # !!! TO DO!!! 
# 
# Make this a bit more automated, but for now, let's also do technical stuff here
#
#
# !!! TO DO!!! # !!! TO DO!!! # !!! TO DO!!! # !!! TO DO!!! # !!! TO DO!!! 


from machine_learning.datasetclass import dataset_classes as cdc
    # import importlib; importlib.reload(cdc)
datadir = segfolder
train_or_test = 'train'

dataset_train = \
    cdc.ImageDataset_tiles(df_metadata, datadir, train_or_test, 
                    img_suffix='_tile_img_enhanced', lbl_suffix='_tile_seg', # img_suffix='_tile_img'; lbl_suffix='_tile_seg_postpr'
                    transform=cdc.augmentation_pipeline_input, 
                    transform_label=cdc.augmentation_pipeline_label, 
                    targetdevice="mps", ARTIFICIAL_N=1000, CROP_SIZE=1000)
dataset_test = \
    cdc.ImageDataset_tiles(df_metadata, datadir, 'test', 
                    img_suffix='_tile_img_enhanced', lbl_suffix='_tile_seg',
                    transform=cdc.augmentation_pipeline_input, 
                    transform_label=cdc.augmentation_pipeline_label, 
                    targetdevice="mps", ARTIFICIAL_N=1000, CROP_SIZE=1000)

### testing

# test if it works
current_img, current_lbl = dataset_train[0]
current_img, current_lbl = dataset_test[0]

# now plot
torch_img = current_img.cpu().squeeze()
img_RGB   = torch_img.permute(1, 2, 0).numpy()
plt.imshow(img_RGB, cmap='gray')
plt.show()

torch_lbl = current_lbl.cpu().squeeze().numpy()
plt.imshow(torch_lbl, cmap='viridis')
plt.show()


# %% ################################################################################
# Now let's initialize the model

from machine_learning.model import unet_model as cunet
    # import importlib; importlib.reload(cunet)

NR_CLASSES=5
modelUNet = cunet.UNet(n_channels=3, n_classes=NR_CLASSES).to("mps")

### testing

# let's create some test input        
X = torch.rand(1, 3, 500, 500, device="mps") # random image as test input
logits = modelUNet(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
print('shape',logits.shape)        
    
# let's test with an actual image
current_img, current_lbl = dataset_train[0]
X = current_img[None, :, :, :] 
logits = modelUNet(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
print('shape',logits.shape)        
    # seems to work OK :)
    
plt.imshow(logits[0,1,:,:].detach().cpu(), cmap='gray')

# %% ################################################################################

from machine_learning.trainer import trainer as ct
    # import importlib; importlib.reload(ct)
    
# Now let's train the network

# libraries
# import importlib; importlib.reload(cdc)

### settings

learning_rate = 1e-3 # 1e-3
BATCH_SIZE = 8

# Custom weights for categories to be used in loss function
label_weights = cdc.get_label_weights(dataset_train, suffix_img='_tile_img_enhanced', suffix_lbl='_tile_seg')
# label_weights = torch.tensor(1/(label_counts/np.sum(label_counts)), dtype=torch.float32).to('mps')
# Use loss function and optimizer geared towards unet (see https://github.com/milesial/Pytorch-UNet)
loss_fn = torch.nn.CrossEntropyLoss(label_weights)  # loss function, weights added MW
optimizer = torch.optim.Adam(modelUNet.parameters(), learning_rate)

# Create data loaders 
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    # shuffle=True also makes sure that batch size is met
    # TEST WHETHER THIS IS CORRECT??!?!
# generator = torch.Generator().manual_seed(42)
val_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# 1 loop execution for testing
# dataloader=train_loader; model=modelCNN
if False:
    #loss_tracker = mt.train_loop(train_loader, modelUNet, loss_fn, optimizer, len(dataset_train), BATCH_SIZE)
    #current_correct = mt.test_loop(val_loader, modelUNet, loss_fn, len(dataset_test), BATCH_SIZE)
    
    # let's see result after test loop
    current_img, current_lbl = dataset_train[1]
    current_img, current_lbl = dataset_test[0]
    X = current_img[None, :, :, :] 
    logits = modelUNet(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    # plot the result
    current_img = current_img.cpu().numpy()
    current_lbl = current_lbl.cpu().numpy()
    current_prd = logits.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    ax[0].set_title('Image')
    ax[0].imshow(current_img[0], cmap='gray')    
    ax[1].set_title('Prediction')    
    ax[1].imshow(current_prd[0].argmax(0), cmap='jet', vmin=0, vmax=4)
    ax[2].set_title('Truth')
    ax[2].imshow(current_lbl, cmap='jet', vmin=0, vmax=4)
    for idx in range(3):
        ax[idx].set_xticks([]); ax[idx].set_yticks([])
    plt.show(); plt.close()
    
    # Overlay prediction on top input
    fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    ax.imshow(current_img[0], cmap='gray')    
    ax.contour((current_prd[0].argmax(0)==1), levels=[.5], colors='red')
    ax.set_xticks([]); ax.set_yticks([])
    plt.show(); plt.close()

    # Also see the different categories
    fig, axs = plt.subplots(1,5)#, figsize=(15*cm_to_inch, 5*cm_to_inch))
    axs.flatten()[0].set_title('Input')
    axs.flatten()[0].imshow(current_img[0], cmap='gray')    
    axs.flatten()[0].set_xticks([]); axs.flatten()[0].set_yticks([])
    for idx in range(4):
        axs.flatten()[idx+1].set_title(['background','body','edge','proximity'][idx])
        axs.flatten()[idx+1].imshow(current_prd[0][idx], cmap='viridis')
        axs.flatten()[idx+1].set_xticks([]); axs.flatten()[idx+1].set_yticks([])
    plt.tight_layout()        
    plt.show(); plt.close()
    
# scheduler = StepLR(optimizer, step_size=20, gamma=0.1) # step_size=10

# define the learning rate during the procedure, using the scheduler
# define fn
def custom_lr_schedule(epoch, step_len=50):
    ''' returns the learning rate scaling factor for the given epoch '''
    lr_scalefactor = [1]*step_len + [.1]*step_len + [.01]*step_len 
    if epoch >= step_len*3:
        epoch = step_len*3-1
    return lr_scalefactor[epoch]    
# define scheduler
#scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: custom_lr_schedule(epoch, step_len=2))
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: custom_lr_schedule(epoch, step_len=8))

# training loop
epochs = 24 # 150 # 30
list_loss_tracker = []
list_correct = []
start_time_overall = time.time()
for t in range(epochs): # t = 0
    
    print('='*30)
    print(f"Epoch {t+1}, LR: {scheduler.get_last_lr()}")
    start_time = time.time()
    
    # train and test
    loss_tracker    = ct.train_loop(train_loader, modelUNet, loss_fn, optimizer, len(dataset_train), BATCH_SIZE)
    current_correct = ct.test_loop(val_loader, modelUNet, loss_fn, len(dataset_test), BATCH_SIZE)
    
    # update scheduler
    scheduler.step()
    
    # track loss & test correctness
    list_loss_tracker.append(loss_tracker)    
    list_correct.append(current_correct)
    
    end_time = time.time()
    elapsed_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Epoch {t+1} completed in {elapsed_time}..")
            
end_time_overall = time.time()
elapsed_time_overall = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time_overall - start_time_overall))
print(f"Training completed in {elapsed_time_overall}..")
print("Done!")

# save the model
current_time_formatted = time.strftime("%Y%m%d_%H%M")
torch.save(modelUNet.state_dict(), modelfolder+'modelUNet'+current_time_formatted+'.pth')

# load the model
if False:
    modelUNet.load_state_dict(torch.load(modelfolder+'modelUNet20251026_1027.pth'))

# %% ################################################################################
# Now some plotting of the results

# Now plot the loss over time
datay = np.array(list_loss_tracker).flatten()
datax = np.array(range(len(datay)))*100
plt.plot(datax, datay, color='black')
plt.ylim([0, np.max(datay)*1.1])
plt.axvline(datax[-1]/3, linestyle='--', color='black')
plt.axvline(datax[-1]/3*2, linestyle='--', color='black')

# and also plot the list_correct
data_correctx = np.linspace(datax[-1]/epochs, datax[-1], epochs)
data_correcty = np.array(list_correct)
plt.plot(data_correctx, data_correcty)

plt.show(); plt.close()

# Let's set a custom color scheme
custom_colors_plantclasses = \
    [   # background black
        '#000000', 
        # shoot light green
        '#90EE90',
        # root white
        '#FFFFFF', 
        # seed brown
        '#A52A2A', 
        # leaf darkgreen
        '#006400' 
        ]
# Create a custom cmap
from matplotlib.colors import ListedColormap
cmap_plantclasses = ListedColormap(custom_colors_plantclasses)

def sidebysideplot(current_img_RGB, current_prd, current_lbl, whichone, pltfolder):

    # make plot
    fig, ax = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    #fig.suptitle(f'Dataset: {whichone}')
    #ax[0].set_title('Image')
    ax[0].imshow(current_img_RGB)
    #ax[1].set_title('Prediction')
    ax[1].imshow(current_prd[0].argmax(0), cmap=cmap_plantclasses, vmin=0, vmax=5)
    #ax[2].set_title('Truth')
    ax[2].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)  
    for idx_ax in range(3):
        ax[idx_ax].set_xticks([]); ax[idx_ax].set_yticks([])
    # plt.show(); plt.close()
    # save it
    plt.tight_layout()
    plt.savefig(pltfolder + f'prediction_{whichone}_{idx:03d}.pdf', dpi=300)

def overlayplot(current_img_RGB, current_prd, current_lbl, whichone, pltfolder):
    # current_img_RGB=img_test_crop_norm, current_prd=img_pred_lbls, current_lbl=np.zeros_like(img_pred_lbls); whichone='fullimage_test'
    
    # apply arg-max if not done yet
    if current_prd.ndim > 2:
        current_pred_lbl = current_prd[0].argmax(0)
    else:
        current_pred_lbl = current_prd
    
    # translate the current_pred_lbl by padding on the left side with zeroes
    current_pred_lbl_transl = current_pred_lbl.copy()
    current_pred_lbl_transl = np.pad(current_pred_lbl, ((0,0),(20,0)), 'constant', constant_values=0)
    
    img_shape_ratio = current_img_RGB.shape[1] / current_img_RGB.shape[0]
    
    # now plot
    if current_lbl is None:
        fig, ax = plt.subplots(1, 1, figsize=(10*img_shape_ratio*cm_to_inch, 10*cm_to_inch))
        ax.imshow(current_img_RGB)    
        ax.imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0*(current_pred_lbl_transl>0))
    else: 
        fig, ax = plt.subplots(1, 2, figsize=((5*img_shape_ratio+5)*cm_to_inch, 5*cm_to_inch))
        ax[0].imshow(current_img_RGB)    
        ax[0].imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0*(current_pred_lbl_transl>0))    
        ax[1].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)  
    
    plt.tight_layout()
    plt.savefig(pltfolder + f'predictionoverlay_{whichone}_{idx:03d}.pdf', dpi=300)
    

# now once again apply the model and show the result
for whichone in ['test', 'train']:
    for idx in range(10):
        
        if whichone=='train':
            current_img, current_lbl = dataset_train[idx]
        elif whichone=='test':
            current_img, current_lbl = dataset_test[idx]
            
        X = current_img[None, :, :, :]
        logits = modelUNet(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
        
        # plot the result
        current_img = current_img.cpu().numpy()
        current_img_RGB = current_img.transpose(1,2,0)
        current_lbl = current_lbl.cpu().numpy()
        current_prd = logits.cpu().detach().numpy()
        
        # now calculate the accuracy
        correct_pixels = (current_prd[0].argmax(0) == current_lbl)
        accuracy = np.sum(correct_pixels) / correct_pixels.size
        print(f'Accuracy for {whichone} image {idx}: {accuracy*100:.2f} %')
        
        sidebysideplot(current_img_RGB, current_prd, current_lbl, whichone, pltfolder)
        overlayplot(current_img_RGB, current_prd, current_lbl, whichone, pltfolder)


# %% ################################################################################
# Now try a full sized data image

image_path_test2 = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/20250721batch1/20250721_OY_09.jpg'

from PIL import Image
img_full = np.array(Image.open(image_path_test2).convert('RGB'))
plt.imshow(img_full); plt.show()

# now normalize the image
img_full_norm = caa.image_autorescale(img_full, rescalelog=False, bg_percentile=10)

# now feed to U-net
with torch.no_grad():
    
    img_full_torch = ToTensor()(img_full_norm).to('mps')
    X = img_full_torch[None, :, :, :]

    # display the final image
    plt.imshow(X[0].cpu().permute(1,2,0).numpy()); plt.show()

    logits = modelUNet(X) # "logits" refers to raw, unnormalized
    prd_full = logits.cpu().detach().numpy()
    
plt.imshow(prd_full[0].argmax(0), cmap=cmap_plantclasses, vmin=0, vmax=4); plt.show()


# now plot as above
current_img = X[0].cpu().numpy()
current_img_RGB = current_img.transpose(1,2,0)
current_prd = prd_full
current_lbl='N_A'
overlayplot(current_img_RGB, current_prd, current_lbl, pltfolder)


# %% ################################################################################
# Now loop over all data;

# import importlib; importlib.reload(crw)

from PIL import Image


import prepostprocessing_input.preprocessing_input as cpp


# First need to do pre-processing
DATA_PATH = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/'

# Now construct a dataframe with all image files (either jpg or tif) in this directory, 
# also listing their subfolder and filename in a column
all_subfolders = glob.glob(pathname='*',  root_dir=DATA_PATH)
crw.gen_metadatafile(DATA_PATH, subdirs=all_subfolders, outputdirectory=outputdirectory, suffix='_inputdata',
                     default_columns = ['comments'])

# 
METADATA_FILEPATH_INPUT = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/metadata_imagefiles_manual_inputdata20251027.xlsx'
# load that file
df_metadata_input = pd.read_excel(METADATA_FILEPATH_INPUT)


def read_input_img_file(df_metadata_input, file_idx):
    
    # now read one image file
    img_test = crw.loadimgfile_metadata(df_metadata_input, file_idx)
    # auto crop
    img_test_crop, _ = cpp.preprocess_getbbox_insideplate(img_test)
    # normalize
    img_test_crop_norm = caa.image_autorescale(img_test_crop, rescalelog=False, bg_percentile=10)

    # plt.imshow(img_test_crop_norm); plt.show()
    
    return img_test_crop_norm


def get_ML_prediction(img_input, themodel, showplot=False):
    # img_input = img_test_crop_norm
            
    # Feed the input image to the U-net
    with torch.no_grad():
        
        img_torch = ToTensor()(img_input).to('mps')
        X = img_torch[None, :, :, :]
            # plt.imshow(X[0].cpu().permute(1,2,0).numpy()); plt.show()

        logits = themodel(X) # "logits" refers to raw, unnormalized
        prd_full     = logits.cpu().detach().numpy()
        prd_labels  = prd_full[0].argmax(0)

    if showplot:        
        plt.imshow(prd_labels, cmap=cmap_plantclasses, vmin=0, vmax=4); plt.show()
    
    return prd_labels


img_test_crop_norm = read_input_img_file(df_metadata_input, file_idx=0)
img_pred_lbls = get_ML_prediction(img_test_crop_norm, themodel=modelUNet, showplot=False)

# now plot as above
overlayplot(img_test_crop_norm, img_pred_lbls, None,
            whichone='fullimage_test', pltfolder=pltfolder)


# now do the 100 random files and store in subfolder of pltfolder
os.makedirs(pltfolder + 'fullimages_predictions/', exist_ok=True)
hundred_file_idxs = np.random.choice(df_metadata_input.index, size=100, replace=False)
for file_idx in hundred_file_idxs:
    
    print(f'Processing file idx {file_idx} ..')
    img_test_crop_norm = read_input_img_file(df_metadata_input, file_idx=file_idx)
    img_pred_lbls = get_ML_prediction(img_test_crop_norm, themodel=modelUNet, showplot=False)
    
    # now plot as above
    print('Plotting prediction')
    overlayplot(img_test_crop_norm, img_pred_lbls, None,
                whichone=f'fullimage_idx{file_idx:03d}', pltfolder=pltfolder + 'fullimages_predictions/')
    
    # also save the predicted mask to a numpy file
    print('Saving prediction..')
    np.save(pltfolder + 'fullimages_predictions/' + f'predictedmask_idx{file_idx:03d}.npy', img_pred_lbls)


# %%
