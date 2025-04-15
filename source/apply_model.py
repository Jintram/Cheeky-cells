

# Load applicable library
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from PIL import Image

import numpy as np

import sys
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2025_Cheeky-cells/')
import source.preprocessing as cheepre

import mypytorch.mymodels as mm
import mypytorch.mytrainer as mt

import seaborn as sns

cm_to_inch = 1/2.54

# add bang_wong_colors vector
color_palette = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000"   # Black
] 
    

def loadmodel():
    
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_lr.1_66pct_2025-04-09_14-33.pth'
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_verySimple-78pct_2025-04-10_17-48.pth'    
    model = mm.VerySimpleNN().to("mps")
    
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_CNN_90pct_2025-04-10_18-16.pth'
    model = mm.CNN((1, 29, 29),6).to("mps")
    
    # The 86% model, trained on reproducibly shuffled data
    PATH = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_CNN_86pct_2025-04-14_15-59.pth'
    model = mm.CNN((1, 29, 29),6).to("mps")
    
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    # sum(p.numel() for p in model.parameters())
    
    return model

def loadimage():
    
    # arbitrary iamge
    filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/resizedTo20_2x Cellen 3 (20x vegroting).jpg'
    # image that was trained on
    filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/resizedTo20_Cellen tellen, 10x foto 1.jpg'
    img = Image.open(filepath)
    img = np.array(img)
    
    # also load the annotation
    filepath_annot = filepath.replace('.jpg', '_seg.npy').replace('_grey','_humanannotated')
    img_annot = np.load(filepath_annot)
    
    return img    
    
def applymodeltoimage_batchmode(img):

    SIZE_HALF=14
    
    image_sizei = img.shape[0]
    image_sizej = img.shape[1]

    start_i = SIZE_HALF+1
    start_j = SIZE_HALF+1
    end_i   = image_sizei - SIZE_HALF
    end_j   = image_sizej - SIZE_HALF

    output_img = np.zeros_like(img, dtype=np.int8)
    #img_collection_tensor = torch.zeros((end_i-start_i)*(end_j-start_j), 1, 29, 29, requires_grad=False).to("mps")
    img_collection_array  = np.zeros([(end_i-start_i)*(end_j-start_j), 1, 29, 29], dtype=np.uint8)
    
    print('Preparing image')
    for idx_i in range(start_i, end_i):
        for idx_j in range(start_j, end_j):
            # idx_i = SIZE_HALF+1; idx_j = SIZE_HALF+1
            
            # define the square
            square = img[(idx_i-SIZE_HALF):(idx_i+SIZE_HALF+1), (idx_j-SIZE_HALF):(idx_j+SIZE_HALF+1)].astype(np.uint8)
            square_exp = np.expand_dims(square, axis=0) # expects shape [1, 29, 29]
            
            # convert to tensor
            # This requires insane amounts of memory for larger objects
            # Need to do this later on the fly
            # square_tensor = torch.from_numpy(square_exp).to("mps")
                # plt.imshow(square); plt.show(); plt.close()
            
            # add tensor to collection
            idx = (idx_i-start_i)*(end_j-start_j) + (idx_j-start_j)
            img_collection_array[idx] = square_exp
        
        if idx_i % 100 == 0:
             print(f'{idx_i}/{end_i} lines done')
           
    print('Creating predictions')
                            
    # apply model at once
    # pred_img = model(img_collection_tensor)
    
    # build up the prediction image in batches of 1000
    batch_size = 1000
    num_batches = int(np.ceil(img_collection_array.shape[0] / batch_size))    
    # flattened pred image with one-hot coding
    pred_img = np.zeros([img_collection_array.shape[0], 6])
    # pred_img = torch.zeros((img_collection_tensor.shape[0], 6), requires_grad=False).to("mps")
    for batch_idx in range(num_batches):
        # batch_idx = 0
        # batch_idx += 1
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, img_collection_array.shape[0])
        
        # apply model
        # DIVIDE BY 255 TO NORMALIZE!
        current_input = torch.from_numpy(img_collection_array[start_idx:end_idx].astype(np.float32)/255).to("mps")
        current_pred = model(current_input)
        
        # save it to applicable location
        # conversion to numpy necessary, otherwise memory heavy (crash)
        pred_img[start_idx:end_idx] = current_pred.cpu().detach().numpy()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx+1}/{num_batches} done')
    
    # apply argmax
    pred_img_int = np.argmax(pred_img, axis=1)
    
    # use reshape to get same dimensions as original image
    output_img = pred_img_int.reshape((end_i-start_i), (end_j-start_j))
    
    
def showsidebyside():
            
    thecategories = ['unclassified', 'background', 'cell border', 'intercell border', 'cell cytoplasm', 'nuclei']
    # convert color_palette to cmap
    cmap = ListedColormap(color_palette[:len(thecategories)])
    
    theunique, thecounts = np.unique(output_img, return_counts=True)
    categories_as_int = np.arange(len(thecategories))
    thebins = np.linspace(0, len(thecategories), len(thecategories)+1)-.5
    thebincounts, _ = np.histogram(output_img, bins=thebins)

    fig, ax = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
    ax[0].imshow(img)
    im = ax[1].imshow(output_img, cmap=cmap, vmin=thebins[0], vmax=thebins[-1])
    # Add color legend
    if False:
        cbar = fig.colorbar(im, ax=ax[1], ticks=categories_as_int, orientation='vertical')
    #cbar.ax.set_yticklabels(thecategories)
    plt.show(); plt.close()  
    
    fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    sns.barplot(x=categories_as_int, y=thebincounts, ax=ax, palette=color_palette)
    for idx in range(len(thecategories)):
        ax.text(idx, max(thecounts)/100, str(thecategories[idx]), color='black', ha='center', va='bottom', rotation=90)
    ax.tick_params(axis='x', rotation=90)    
    ax.set_xlabel('Category'); ax.set_ylabel('Times observed')
    plt.tight_layout()
    plt.show(); plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    sns.barplot(y=categories_as_int, x=thebincounts, ax=ax, palette=color_palette, orient='h')
    for idx in range(len(thecategories)):
        ax.text(.2, idx, str(thecategories[idx]), color='black', ha='left', va='center', rotation=0)
    ax.tick_params(axis='y', rotation=0)    
    ax.set_ylabel('Category'); ax.set_xlabel('Times observed')
    # set log scale for x
    ax.set_xscale('log')
    ax.set_xlim([.1, np.max(thebincounts)*2])
    plt.tight_layout()
    plt.show(); plt.close()
    
    # create alpha map for img_annot
    thealphamap = np.zeros_like(img_annot, dtype=np.float32)
    thealphamap[img_annot>0] = 1
    
    # now also show the ground truth, img_annot
    fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch, 5*cm_to_inch))
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.imshow(img_annot, cmap=cmap, vmin=thebins[0], vmax=thebins[-1], alpha=thealphamap)
    #cbar.ax.set_yticklabels(thecategories)
    # save to desktop
    #plt.savefig('/Users/m.wehrens/Desktop/overlay.pdf', dpi=300)
    plt.show(); plt.close()  
    
    
    
    # histogram of the unique values in output_img
    
    counts
    

def sanitycheckwithdata():
    '''
    Apply the model to the dataset that was used for training and testing.
    '''
    
    import importlib; importlib.reload(cheepre)
    
    list_all_images_collected, complete_label_table_sel  = cheepre.return_testset_array_manual([1]) 
    # add 1 additional axis to list_all_images_collected
    list_all_images_collected = np.array(list_all_images_collected)
    list_all_images_collected = np.expand_dims(list_all_images_collected, axis=1)
        
    testimgs_torch_array = torch.from_numpy(list_all_images_collected.astype(np.float32)).to("mps")
    
    # show a single image
    img_forshow = list_all_images_collected[2,0,:,:]
    img_forshow = testimgs_torch_array[2,0,:,:].detach().to("cpu").numpy()
    plt.imshow(img_forshow); plt.show(); plt.close()
    complete_label_table_sel[3,2]
    
    # Let's try to generate random data and see what happens
    # testimgs_torch_array = torch.rand(50, 1, 29, 29, device="mps") # random image as test input
    
    # now apply the model to testimgs_torch_array
    current_pred = model(testimgs_torch_array).detach().to("cpu").numpy()
        
    # apply argmax
    pred_img_int = np.argmax(current_pred, axis=1)
    
    # actual labels
    complete_label_table_sel[3,:]
    
def sanitycheckwithdatasetclass():
    
    # Get the dataset
    ANNOT_DIR = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_humanannotated/'
    IMG_DIR   = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/'
    mydataset = md.CustomImageDataset(annot_dir=ANNOT_DIR, 
                                img_dir=IMG_DIR, 
                                transform=ToTensor(), 
                                target_transform=Lambda(lambda y: torch.zeros(6, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                preload_imgs=True)
    
    total_size = len(mydataset)
    train_size = int(0.8 * total_size)  # 80% for training

    # Split the dataset manually, simply taking the first train_size samples
    train_dataset = torch.utils.data.Subset(mydataset, range(train_size))
    val_dataset = torch.utils.data.Subset(mydataset, range(train_size, total_size))

    # Also subset the weights
    train_weights  = mydataset.labels[:train_size]
    val_weights    = mydataset.labels[train_size:total_size]
    
    # set samplers
    sampler_train = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)
    sampler_val   = WeightedRandomSampler(val_weights,   num_samples=len(val_dataset),   replacement=True)

    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, 
    #                        sampler=sampler_train) # untested

    generator = torch.Generator().manual_seed(42)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # get the first 64 samples as a test set
    X, y = next(iter(val_loader))
    pred = model(X)
    pred_img_int = pred.argmax(1).to('cpu').numpy()
    truth_loader = y.argmax(1).to('cpu').numpy()
    pct_correct = np.sum(pred_img_int==truth_loader)/64
    
    ###
    # now get those same samples using the dataset object
    sanity_img_idxs = mydataset.img_idxs[train_size:total_size]
    sanity_posis = mydataset.posis[train_size:total_size]
    sanity_posjs = mydataset.posjs[train_size:total_size]
    sanity_labels = mydataset.labels[train_size:total_size]
    truth_samples = sanity_labels[:64]
    
    # sanity check labels
    np.all(truth_loader==truth_samples)

    SAMPLES_TO_RETURN = 64
    list_all_images_collected = np.zeros([SAMPLES_TO_RETURN, 29, 29], dtype=np.float32)

    for img_idx in range(SAMPLES_TO_RETURN):
    # img_idx = 0
                    
        filename_img = list_allimgpaths[sanity_img_idxs[img_idx]]
        posi=sanity_posis[img_idx]
        posj=sanity_posjs[img_idx]
                                
        img_crop = provide_crop(input_folder=input_folder, 
                        filename_img = filename_img,
                        posi=posi, 
                        posj=posj)                

        list_all_images_collected[img_idx,:,:] = img_crop/255
    
    # now convert this to tensor
    X_man = np.expand_dims(list_all_images_collected, axis=1)
    X_man = torch.from_numpy(X_man.astype(np.float32)).to("mps")
    X_man_0 = X_man[0,0,:,:]
    X_0 = X[0,0,:,:]
    X_man_0==X_0
    
    X_man_0[:10,0]/255
    X_0[:10,0]
    
    # now apply model
    pred_man = model(X)
    # convert to numpy
    pred_man = pred_man.cpu().detach().numpy()
    # apply argmax
    pred_img_int_man = np.argmax(current_pred, axis=1)
    
######### OLD


def applymodeltoimageOLD(img, SIZE_HALF=14):
    
    image_sizei = img.shape[0]
    image_sizej = img.shape[1]

    start_i = SIZE_HALF+1
    start_j = SIZE_HALF+1
    end_i   = image_sizei - SIZE_HALF
    end_j   = image_sizej - SIZE_HALF

    output_img = np.zeros_like(img, dtype=np.int8)
    for idx_i in range(start_i, end_i):
        for idx_j in range(start_j, end_j):
            # idx_i = SIZE_HALF+1; idx_j = SIZE_HALF+1
            
            # define the square
            square = img[(idx_i-SIZE_HALF):(idx_i+SIZE_HALF+1), (idx_j-SIZE_HALF):(idx_j+SIZE_HALF+1)].astype(np.float32)
            square_exp = np.expand_dims(square, axis=0) # expects shape [1, 29, 29]
            
            # convert to tensor
            square_tensor = torch.from_numpy(square_exp).to("mps")
                # plt.imshow(square); plt.show(); plt.close()
                        
            # apply model
            pred = model(square_tensor)
            # get the index of the max value
            pred_idx = torch.argmax(pred, dim=1)
            # convert to python integer
            pred_idx = int(pred_idx.item())
            
            output_img[idx_i, idx_j] = pred_idx
            
        if ((idx_i) % 10) == 0:
            print(f'{idx_i}/{end_i} lines done ({round(idx_i/end_i*100,1)}%).')
            unique, counts = np.unique(output_img, return_counts=True)
            print(counts)
    