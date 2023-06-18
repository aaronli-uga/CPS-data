import numpy as np
import copy
import torch
import torch.nn as nn 
import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def model_train_detection(model, train_loader, val_loader, num_epochs, optimizer, device, history):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        num_epochs (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        history (_type_): the statistic information such as accuracy, f1-score, loss

    Returns:
        _type_: statistical information. history
    """    
    loss_fn = nn.BCELoss()
    
    best_f1 = -np.inf
    best_weights = None
    
    for epoch in range(num_epochs):
        train_acc, train_loss = [], []
        model.train()
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
            # forward pass
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = (y_pred.round() == y_batch).float().mean()
            
            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()
        
        # evaluate at end of each epoch
        model.eval()
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.float32)
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            f1 = f1_score(y_true=y_val.cpu().detach().numpy(), 
                          y_pred=y_pred.round().cpu().detach().numpy(), 
                          average='binary')
            
            history['train_loss'].append(sum(train_loss) / len(train_loss))
            history['train_acc'].append(sum(train_acc) / len(train_acc))
            history['val_acc'].append(float(acc))
            history['val_f1'].append(float(f1))
            print(f"train acc: {history['train_acc'][-1]}. val acc: {history['val_acc'][-1]}")
            print(f"train loss: {history['train_loss'][-1]}")
            print(f"val f1: {history['val_f1'][-1]}")
            
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = copy.deepcopy(model.state_dict())
                
            # only validate one batch
            break
    
    print(f"Training Done. best F1 score: {best_f1}")
    # save the model with best performance
    model.load_state_dict(best_weights)
    
    return 0


def model_train_diagnosis(model, train_loader, val_loader, num_epochs, optimizer, device, history):
    """_summary_
        model training function for multi-class classification
    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        num_epochs (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        history (_type_): the statistic information such as accuracy, f1-score, loss

    Returns:
        _type_: statistical information. history
    """
    
    loss_fn = nn.CrossEntropyLoss()
    
    best_f1 = -np.inf
    best_weights = None
    
    for epoch in range(num_epochs):
        train_loss, predicted_labels, ground_truth_labels = [], [], []
        model.train()
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
            # forward pass
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            _, predicted = torch.max(y_pred, 1) # get the index of the predicated label, which represent the predicted class number
            predicted_labels.append(predicted.cpu().detach().numpy())
            ground_truth_labels.append(y_batch.cpu().detach().numpy())
            # store metrics
            train_loss.append(float(loss))            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        
        # evaluate at end of each epoch
        model.eval()
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.long)
            y_pred = model(X_val)
            loss = loss_fn(y_pred, y_val)
            _, predicted = torch.max(y_pred, 1)
            predicted = predicted.cpu().detach().numpy()
            y_val = y_val.cpu().detach().numpy()
            
            val_acc = accuracy_score(predicted, y_val)
            val_f1 = f1_score(predicted, y_val, average='macro')
            val_f1_none = f1_score(predicted, y_val, average=None)
            
            predicted_labels = np.concatenate(predicted_labels).ravel()
            ground_truth_labels = np.concatenate(ground_truth_labels).ravel()
            
            history['train_loss'].append(sum(train_loss) / len(train_loss))
            history['val_loss'].append(float(loss))
            history['train_acc'].append(accuracy_score(predicted_labels, ground_truth_labels))
            history['train_f1'].append(f1_score(predicted_labels, ground_truth_labels, average='weighted'))
            history['val_acc'].append(float(val_acc))
            history['val_f1'].append(float(val_f1))
            history['val_f1_all'].append(val_f1_none.tolist())
            
            print("-------------------------------------------")
            print(f"train loss: {history['train_loss'][-1]}. val loss: {history['val_loss'][-1]}\n")
            print(f"train acc: {history['train_acc'][-1]}. val acc: {history['val_acc'][-1]}")
            print(f"train f1: {history['train_f1'][-1]}. val f1: {history['val_f1'][-1]}")
            print(f"val f1 all: {history['val_f1_all'][-1]}")
            
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())
                
            # only validate one batch
            break
    
    print(f"Training Done. best F1 score: {best_f1}")
    # save the model with best performance
    model.load_state_dict(best_weights)
    
    return 0