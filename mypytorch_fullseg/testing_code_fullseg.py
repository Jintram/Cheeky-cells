

def examplecode():
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import mypytorch_fullseg.mytrainer_fullseg as mt
    import mypytorch_fullseg.dataset_classes_fullseg as md
    
    import mypytorch_fullseg.models_downloaded.unet_model as um
    
    import importlib; importlib.reload(mt); importlib.reload(md); importlib.reload(um)

    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import ToTensor, Lambda
    from torch.optim.lr_scheduler import StepLR, LambdaLR
    
    import torch
    import time    
    
    cm_to_inch = 1/2.54

    # test loading the data, without transformer
    DIR_SAVE_MODELS = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/UNET_MODELS/'
    ANNOT_DIR='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    METADATA_FILE='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/metadata_Fluoppi_data20250328_MACHINELEARN.xlsx'
    SIZE_ARTIFICIAL = 1000; BATCH_SIZE = 8
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
        loss_tracker = mt.train_loop(train_loader, modelUNet, loss_fn, optimizer, len(mydataset_train), BATCH_SIZE)
        current_correct = mt.test_loop(val_loader, modelUNet, loss_fn, len(mydataset_test), BATCH_SIZE)
        
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
    plt.plot(datax, datay)
    plt.ylim([0, np.max(datay)*1.1])
    plt.axvline(datax[-1]/3)
    plt.axvline(datax[-1]/3*2)

    # and also plot the list_correct
    data_correctx = np.linspace(datax[-1]/epochs, datax[-1], epochs)
    data_correcty = np.array(list_correct)
    plt.plot(data_correctx, data_correcty)
    
    plt.show(); plt.close()
    