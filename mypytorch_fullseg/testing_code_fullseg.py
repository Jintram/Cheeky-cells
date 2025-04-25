

def examplecode():
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import mypytorch_fullseg.mytrainer_fullseg as mt
    import mypytorch_fullseg.dataset_classes_fullseg as md
    
    import mypytorch_fullseg.models_downloaded.unet_model as um
    
    import importlib; importlib.reload(mt); importlib.reload(md); importlib.reload(um)

    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import ToTensor, Lambda
    from torch.optim.lr_scheduler import StepLR

    import torch
    import time    
    
    cm_to_inch = 1/2.54

    # test loading the data, without transformer
    ANNOT_DIR='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    METADATA_FILE='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/metadata_Fluoppi_data20250328_MACHINELEARN.xlsx'
    SIZE_ARTIFICIAL = 1000
    mydataset_test = md.ImageDataset_tiles(annot_dir=ANNOT_DIR, metadata_file=METADATA_FILE, train_or_test='test', 
                                           transform = md.augmentation_pipeline_input, 
                                           transform_target=md.augmentation_pipeline_target, 
                                            targetdevice="mps", SIZE_ARTIFICIAL = SIZE_ARTIFICIAL)
    mydataset_train = md.ImageDataset_tiles(annot_dir=ANNOT_DIR, metadata_file=METADATA_FILE, train_or_test='train', 
                                           transform = md.augmentation_pipeline_input, 
                                           transform_target=md.augmentation_pipeline_target, 
                                            targetdevice="mps", SIZE_ARTIFICIAL = SIZE_ARTIFICIAL)
    
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
        
    current_img = current_img.cpu().numpy()
    current_lbl = current_lbl.cpu().numpy()
    current_prd = logits.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15*cm_to_inch, 5*cm_to_inch))
    ax[0].set_title('Image')
    ax[0].imshow(current_img[0], cmap='gray')    
    ax[1].set_title('Prediction')    
    ax[1].imshow(current_prd[0].argmax(0), cmap='gray')
    ax[2].set_title('Label')
    ax[2].imshow(current_lbl, cmap='gray')    
    plt.show()
            
    ### settings
    
    learning_rate = 1e-1 # 1e-3

    # From UNet
    loss_fn = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(modelUNet.parameters(), learning_rate)
    
    # Create data loaders
    BATCH_SIZE = 16
    train_loader = DataLoader(mydataset_train, batch_size=BATCH_SIZE, shuffle=True)
        # shuffle=True also makes sure that batch size is met
        # TEST WHETHER THIS IS CORRECT??!?!
    # generator = torch.Generator().manual_seed(42)
    val_loader = DataLoader(mydataset_test, batch_size=BATCH_SIZE, shuffle=True)

    # 1 loop execution for testing
    # dataloader=train_loader; model=modelCNN
    if False:
        loss_tracker = mt.train_loop(train_loader, modelUNet, loss_fn, optimizer, len(mydataset_train), BATCH_SIZE)
        current_correct = mt.test_loop(val_loader, modelUNet, loss_fn, len(mydataset_test), BATCH_SIZE)
        
        # let's see result after test loop
        current_img, current_lbl = mydataset_train[0]
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
        ax[1].imshow(current_prd[0].argmax(0), cmap='gray')
        ax[2].set_title('Label')
        ax[2].imshow(current_lbl, cmap='gray')    
        plt.show()        
        
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1) # step_size=10

    epochs = 9 # 30
    list_loss_tracker = []
    list_correct = []
    start_time_overall = time.time()
    for t in range(epochs):
        
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