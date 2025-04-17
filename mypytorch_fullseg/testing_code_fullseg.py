



def examplecode():

    import mypytorch_fullseg.mytrainer as mt
    import mypytorch_fullseg.dataset_classes as md
    import mypytorch_fullseg.mymodels as mm
    import importlib; importlib.reload(mt); importlib.reload(md); importlib.reload(mm)

    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import ToTensor, Lambda

    import torch

    modelCNN = mm.CNN((1, 29, 29),6).to("mps")
        # sum(p.numel() for p in modelCNN.parameters())
        
    X = torch.rand(1, 1, 29, 29, device="mps") # random image as test input
    logits = modelCNN(X) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    print('shape',logits.shape)        
        
    # Define model
    ANNOT_DIR = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_humanannotated/'
    IMG_DIR   = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/'
    mydataset = md.CustomImageDataset(annot_dir=ANNOT_DIR, 
                                img_dir=IMG_DIR, 
                                transform=ToTensor(), 
                                target_transform=Lambda(lambda y: torch.zeros(6, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        
    # Split the dataset
    # Define the sizes of the splits
    total_size = len(mydataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size   # 20% for validation
    # Split the dataset manually, simply taking the first train_size samples
    train_dataset = torch.utils.data.Subset(mydataset, range(train_size))
    val_dataset = torch.utils.data.Subset(mydataset, range(train_size, total_size))
        
    
        
        
    image, label = mydataset[0]        
    image.shape
        
    ### settings
    
    learning_rate = 1e-1 # 1e-3

    # From CNN   
    loss_fn = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(modelCNN.parameters(), learning_rate)
    
    TOTAL_SAMPLES     = int(100_000)
    TOTAL_TESTSAMPLES = int(10_000)
    BATCH_SIZE        = 64
    NUM_TESTBATCHES   = int(np.ceil(TOTAL_TESTSAMPLES/64))


    # get the first 10 samples in mydataset
    for i in range(10):
        print(mydataset[i][0].shape, mydataset[i][1])
    # create a subset of mydataset, mydataset_subset, with only the first 10 samples
    mydataset_subset = torch.utils.data.Subset(mydataset, range(100_000))
    # create a subset based on the vector SELECTION
    SELECTION = np.array([True, True, True, False, False, False, False, False, False, False])
    mydataset_subset = torch.utils.data.Subset(mydataset, np.where(SELECTION)[0])
    


    #########
    # Split the dataset        
    from torch.utils.data import random_split

    # Define the sizes of your splits
    total_size = len(mydataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size   # 20% for validation

    # Create the splits
    train_dataset, val_dataset = random_split(
        mydataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    #########
    # Add weights to the dataset
    original_labels = mydataset.labels
    labels_trainset    = original_labels[train_dataset.indices]
    labels_valset      = original_labels[val_dataset.indices]

    # weights for training set
    train_bincounts = np.bincount(labels_trainset)
    class_weights_train = 1.0 / train_bincounts
    class_weights_train[train_bincounts==0] = 0 # counters div 0
    weights_train = class_weights_train[labels_trainset]  # Assign a weight to each sample
    # also for val_set
    val_bincounts = np.bincount(labels_valset)
    class_weights_val = 1.0 / val_bincounts
    class_weights_val[val_bincounts==0] = 0
    weights_val = class_weights_val[labels_valset]  # Assign a weight to each sample
    # set samplers
    sampler_train = WeightedRandomSampler(weights_train, num_samples=len(train_dataset), replacement=True)
    sampler_val   = WeightedRandomSampler(weights_val, num_samples=len(val_dataset), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, 
                            sampler=sampler_train) # untested

    generator = torch.Generator().manual_seed(42)
    val_loader = DataLoader(val_dataset, batch_size=64, # shuffle=False, 
                            shuffle=False, generator=generator, # untested
                            sampler=sampler_val)


    # 1 loop execution
    # dataloader=train_loader; model=modelCNN
    loss_tracker = mt.train_loop(train_loader, modelCNN, loss_fn, optimizer, TOTAL_SAMPLES=TOTAL_SAMPLES, BATCH_SIZE=BATCH_SIZE)
    current_correct  = mt.test_loop(val_loader, modelCNN, loss_fn, NUM_TESTBATCHES=NUM_TESTBATCHES)

    CONTINUE BY TESTING THE ABOVE FUNCTIONS, WHETHER THEY OPERATE PROPERLY
    (HAD ALREADY STARTED).