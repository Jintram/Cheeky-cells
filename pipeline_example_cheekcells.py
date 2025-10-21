

# %% ################################################################################
# Some settings

SCRIPT_DIR = '/Users/m.wehrens/Documents/git_repos/_UVA/2025_Cheeky-cells/'

# Directory with image files
basedirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/'
subdirs = ['Cheek-Cells_resized_anonymous/']

# Directory to place output
outputdirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/'

# add scripts in this folder to path
import sys
if SCRIPT_DIR not in sys.path: sys.path.append(SCRIPT_DIR)

# %% ################################################################################

# standard libs
import pandas as pd
import matplotlib.pyplot as plt
import torch


# custom libs
import readwrite.cheeky_readwrite as crw
    # reload
    # import importlib; importlib.reload(crw)

import annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
    
import annotating_data.annotation_postprocessing as cap
    # import importlib; importlib.reload(cap)
    


# %% ################################################################################

# First generate the metadata file
crw.gen_metadatafile(basedirectory, subdirs, outputdirectory)
    # This will auto-generate a metadata file with all image files in the specified folder.
    # It should be manually edited, and stored as an excel file (at metadata_filepath)

# Now update the metadata file location, after editing the auto-generated file and saving under custom name
metadata_customized_filename = 'metadata_imagefiles_MW2025oct-SELECTION.xlsx'
metadata_filepath            = outputdirectory + metadata_customized_filename

# Now load the (manually updated) metadata
df_metadata = pd.read_excel(metadata_filepath)

# %% ################################################################################

# Loop over the files to create the ground truth 
caa.annotate_all_pictures_aided(df_metadata, 
                                output_segfolder=outputdirectory + 'humanseg/',
                                intitial_segfolder=None,
                                TILE_SIZE=1000,
                                tile_selection_by='maxvar',
                                segfn=caa.basicseg2,
                                ignore_saved_file=False,
                                showplots=False,
                                rescalelog=False)


# %% ################################################################################

# Now let's post-process the segmented files to get an improved training mask
segfolder = outputdirectory + 'humanseg/'

# Improve all basic segmentations, ie add boundaries proximity zones
cap.postprocess_basicsegfile_all(df_metadata, segfolder, suffix='_seg_postpr', showplot=True)

# %% ################################################################################
# Now set up a dataset, model, and start training

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
                    img_suffix='_tile_img', lbl_suffix='_tile_seg_postpr', # img_suffix='_tile_img'; lbl_suffix='_tile_seg_postpr'
                    transform=cdc.augmentation_pipeline_input, 
                    transform_label=cdc.augmentation_pipeline_label, 
                    targetdevice="mps", ARTIFICIAL_N=1000, CROP_SIZE=1000)
dataset_test = \
    cdc.ImageDataset_tiles(df_metadata, datadir, 'test', 
                    img_suffix='_tile_img', lbl_suffix='_tile_seg_postpr',
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

modelUNet = cunet.UNet(n_channels=3, n_classes=4).to("mps")

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

# Now let's train the network


XXXXXXXXXXXXXXXX


### settings

learning_rate = 1e-3 # 1e-3

# Custom weights for categories to be used in loss function
label_counts  = md.get_label_frequencies_train(METADATA_FILE, ANNOT_DIR)
label_weights = torch.tensor(1/(label_counts/np.sum(label_counts)), dtype=torch.float32).to('mps')
# Use loss function and optimizer geared towards unet (see https://github.com/milesial/Pytorch-UNet)
loss_fn = torch.nn.CrossEntropyLoss(label_weights)  # loss function, weights added MW
optimizer = torch.optim.Adam(modelUNet.parameters(), learning_rate)

# Create data loaders    
train_loader = DataLoader(mydataset_train, batch_size=BATCH_SIZE, shuffle=True)
    # shuffle=True also makes sure that batch size is met
    # TEST WHETHER THIS IS CORRECT??!?!
# generator = torch.Generator().manual_seed(42)
val_loader = DataLoader(mydataset_test, batch_size=BATCH_SIZE, shuffle=True)

# 1 loop execution for testing
# dataloader=train_loader; model=modelCNN
if False:
    #loss_tracker = mt.train_loop(train_loader, modelUNet, loss_fn, optimizer, len(mydataset_train), BATCH_SIZE)
    #current_correct = mt.test_loop(val_loader, modelUNet, loss_fn, len(mydataset_test), BATCH_SIZE)
    
    # let's see result after test loop
    current_img, current_lbl = mydataset_train[1]
    current_img, current_lbl = mydataset_test[0]
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
def custom_lr_schedule(epoch):
    ''' returns the learning rate scaling factor for the given epoch '''
    lr_scalefactor = [1]*50 + [.1]*50 + [.01]*50 
    if epoch >= 150:
        epoch = 149
    return lr_scalefactor[epoch]    
# define scheduler
scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_schedule)

# training loop
epochs = 150 # 30
list_loss_tracker = []
list_correct = []
start_time_overall = time.time()
for t in range(epochs):
    
    print('='*30)
    print(f"Epoch {t+1}, LR: {scheduler.get_last_lr()}")
    start_time = time.time()
    
    # train and test
    loss_tracker    = mt.train_loop(train_loader, modelUNet, loss_fn, optimizer, len(mydataset_train), BATCH_SIZE)
    current_correct = mt.test_loop(val_loader, modelUNet, loss_fn, len(mydataset_test), BATCH_SIZE)
    
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
torch.save(modelUNet.state_dict(), DIR_SAVE_MODELS+'modelUNet'+current_time_formatted+'.pth')

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


