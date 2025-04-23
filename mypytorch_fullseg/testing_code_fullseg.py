



def examplecode():
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import mypytorch_fullseg.mytrainer_fullseg as mt
    import mypytorch_fullseg.dataset_classes_fullseg as md
    
    import mypytorch_fullseg.models_downloaded.unet_model as um
    
    import importlib; importlib.reload(mt); importlib.reload(md); importlib.reload(um)

    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import ToTensor, Lambda

    import torch

    # test loading the data, without transformer
    ANNOT_DIR='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    METADATA_FILE='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/metadata_Fluoppi_data20250328_MACHINELEARN.xlsx'
    mydataset_test = md.ImageDataset_tiles(annot_dir=ANNOT_DIR, metadata_file=METADATA_FILE, train_or_test='test', 
                                           transform = md.augmentation_pipeline_input, 
                                           transform_target=md.augmentation_pipeline_target, 
                                            targetdevice="mps")
    mydataset_train = md.ImageDataset_tiles(annot_dir=ANNOT_DIR, metadata_file=METADATA_FILE, train_or_test='train', 
                                           transform = md.augmentation_pipeline_input, 
                                           transform_target=md.augmentation_pipeline_target, 
                                            targetdevice="mps")
    
    # Load and display the first datapoint, for testing
    current_img, current_lbl = mydataset_test[0]
    current_img, current_lbl = mydataset_train[0]
    
    # Show side to side
    current_img = current_img.cpu().numpy()
    current_lbl = current_lbl.cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(current_img[0], cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(current_lbl, cmap='gray')
    ax[1].set_title('Label')
    plt.show()

    # define the model
    modelUNet = um.UNet(n_channels=1, n_classes=4).to("mps")

    # let's create some test input        
    X = torch.rand(1, 1, 500, 500, device="mps") # random image as test input
    logits = modelUNet(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    print('shape',logits.shape)        
        
    # let's test with an actual image
    current_img, current_lbl = mydataset_train[0]
    X = current_img[None, :, :, :] 
    logits = modelUNet(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    print('shape',logits.shape)        
        # seems to work OK :)
        
    # plot the logits
    if False:
        logits = logits.cpu().detach().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.imshow(logits[0][0]); plt.show(); plt.close()
            
    ### settings
    
    learning_rate = 1e-1 # 1e-3

    # From UNet
    loss_fn = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(modelUNet.parameters(), learning_rate)
    
    # Create data loaders
    train_loader = DataLoader(mydataset_train, batch_size=64, shuffle=True)
        # shuffle=True also makes sure that batch size is met
        # TEST WHETHER THIS IS CORRECT??!?!
    # generator = torch.Generator().manual_seed(42)
    val_loader = DataLoader(mydataset_test, batch_size=64, shuffle=True)


    # 1 loop execution
    # dataloader=train_loader; model=modelCNN
    !!FIX THIS CODE, AND THEN ALSO SPIN UP A REAL LOOP!!
    - BATCH SIZES ARE NOT SET CORRECTLY
    - THERE IS SOME ISSUE WITH GETTING 64 SAMPLES BACK..
    loss_tracker = mt.train_loop(train_loader, modelUNet, loss_fn, optimizer, TOTAL_SAMPLES=TOTAL_SAMPLES, BATCH_SIZE=BATCH_SIZE)
    current_correct  = mt.test_loop(val_loader, modelUNet, loss_fn, NUM_TESTBATCHES=NUM_TESTBATCHES)

