
import torch
import numpy as np
import math 

def train_loop(dataloader, model, loss_fn, optimizer, dataset_len, BATCH_SIZE):
    # dataloader=train_loader; model=modelUNet
    
    loss_tracker = []
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # batch = 0; (X, y) = next(iter(dataloader))
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            loss_tracker.append(loss)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_len:>5d}]")
    
    print("Epoch done..")
    
    return loss_tracker
            
            
            
def test_loop(dataloader, model, loss_fn, dataset_len, BATCH_SIZE):
    # dataloader = val_loader; dataset_len = len(mydataset_test)
    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss, correct = 0, 0
    
    print('Predicting for test set.')
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    nr_batches = math.ceil(dataset_len / BATCH_SIZE)
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            # batch_idx = 0; (X, y) = next(iter(dataloader))
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{nr_batches}')
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += ((y == pred.argmax(1)).type(torch.float).sum().item()) / (np.prod(y.shape))

    test_loss /= nr_batches
    correct /= dataset_len
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return correct
