from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import gc

def train_epoch(dataloader: DataLoader, model: nn.Module, optimizer, scheduler=None, device=torch.device("cpu")):
    model.train()
    loss = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    dataset_size = 0
    running_loss = torch.tensor(0.0,device=device)
    for step, data in enumerate(dataloader):
        optimizer.zero_grad()

        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)

        batch_size = ids.size(0)
        with torch.cuda.amp.autocast():
          outputs = model(ids, mask)
          lss = loss(outputs.logits, targets)

        scaler.scale(lss).backward()
        scaler.step(optimizer)

        scaler.update()

        running_loss += lss
        dataset_size += 1

        if scheduler:
            scheduler.step()

    final_loss = running_loss.item() / dataset_size
    gc.collect()
    return final_loss
    
@torch.no_grad()
def validate_model(model,dataloader,device=torch.device("cpu")):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    loss = nn.CrossEntropyLoss()
    
    for step, data in enumerate(dataloader):
        
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)
        
        batch_size = ids.size(0)
        
        outputs = model(ids, mask)
        lss = loss(outputs.logits, targets)

        running_loss += (lss.item() * batch_size)
        dataset_size += batch_size
    
    gc.collect()
        
    return running_loss / dataset_size
    
