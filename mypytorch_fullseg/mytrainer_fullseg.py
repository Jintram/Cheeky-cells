
import torch


def train_loop(dataloader, model, loss_fn, optimizer, TOTAL_SAMPLES, BATCH_SIZE):
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            loss_tracker.append(loss)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/{TOTAL_SAMPLES:>5d}]")
            
        if (batch * BATCH_SIZE + len(X)) > TOTAL_SAMPLES: # 100_000:
            print('Epoch done')
            break
        
    return loss_tracker
            
            
            
def test_loop(dataloader, model, loss_fn, NUM_TESTBATCHES):
    # dataloader = val_loader
    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = 3 # FIX THIS??!?
    num_batches = NUM_TESTBATCHES # len(dataloader)
    test_loss, correct = 0, 0
    
    print('Predicting for test set.')
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            # batch_idx = 0; (X, y) = next(iter(dataloader))
            
            if batch_idx>num_batches:
                break
            if batch_idx % 25 == 0:
                print(f'Batch {batch_idx}')
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += ((y == pred.argmax(1)).type(torch.float).sum().item()) / (np.prod(y.shape))

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return correct
