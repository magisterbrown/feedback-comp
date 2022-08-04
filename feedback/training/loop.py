from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def train_epoch(dataloader: DataLoader, model: nn.Module, optimizer):
    model.train()
    device = torch.device("cpu")
    loss = nn.CrossEntropyLoss()
    
    dataset_size = 0
    running_loss = 0.0
    
    for step, data in enumerate(dataloader):
        
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)
        
        batch_size = ids.size(0)
        
        outputs = model(ids, mask)
        lss = loss(outputs.logits, targets)
        lss.backward()
        optimizer.step()
        
        running_loss += (lss.item() * batch_size)
        dataset_size += batch_size
        
    return running_loss / dataset_size
    
