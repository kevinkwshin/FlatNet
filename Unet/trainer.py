from progressbar import Bar
import torch
import os
import shutil

        
def train(train_loader, model,criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    history_dict = {'train':{'acc':[],'loss':[]},'val':{'acc':[],'loss':[]}}

    for batch_idx, sample in enumerate(Bar(train_loader)):
        
        inputs,targets = sample['image'].to(device), sample['landmarks'].to(device) 
        
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs= torch.sigmoid(outputs)
        loss = criterion(outputs, targets.float())
        #loss = dice_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = train_loss /len(train_loader)
    epoch_acc = correct / total


    print('train | Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))



def val(val_loader, model, criterion, device, best_acc_wts):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(Bar(val_loader)):
            inputs,targets = sample['image'].to(device), sample['landmarks'].to(device) 
            outputs = model(inputs)
            outputs= torch.sigmoid(outputs)
            #loss = criterion(outputs, targets.long()) 
            loss = criterion(outputs, targets.float())
            #loss = dice_loss(outputs, targets)
            test_loss += loss.data.cpu().numpy()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    epoch_loss = test_loss / len(val_loader)
    epoch_acc = correct / total

    print('val | Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))

    return best_acc_wts,epoch_loss        
        
        
        
        
        
        
