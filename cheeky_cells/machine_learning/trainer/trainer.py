#%% ######################################################################

import torch
import numpy as np
import math 

def train_loop(dataloader, model, loss_fn, optimizer, dataset_len, BATCH_SIZE, train_log_interval=10):
    # dataloader=train_loader; model=modelUNet
    
    loss_tracker = []
    
    # Set the model to training mode 
    # (activate dropout and batchnorm)
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

        if batch % train_log_interval == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            loss_tracker.append(loss)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_len:>5d}]")
    
    print("Epoch done..")
    
    return loss_tracker
            
            
            
def test_loop(dataloader, model, loss_fn, dataset_len, BATCH_SIZE, nr_classes=None):
    # dataloader = val_loader; dataset_len = len(mydataset_test)
    # arabidopsis
    # dataloader=val_loader; model= modelUNet; dataset_len=len(dataset_test)
    
    # Set the model to evaluation mode 
    # (no dropout and batchnorm)
    model.eval()
    test_loss, correct = 0, 0

    # Confusion matrix accumulator (allocated once nr_classes is known)
    confusion_matrix = None
    
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

            # Accumulate confusion matrix
            pred_labels = pred.argmax(1).cpu().numpy().flatten()
            true_labels = y.cpu().numpy().flatten()
            n_cls = nr_classes if nr_classes is not None else int(pred.shape[1])
            if confusion_matrix is None:
                confusion_matrix = np.zeros((n_cls, n_cls), dtype=np.int64)
            np.add.at(confusion_matrix, (true_labels, pred_labels), 1)

    test_loss /= nr_batches
    correct /= nr_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return correct, test_loss, confusion_matrix
