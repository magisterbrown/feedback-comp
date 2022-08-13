import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from feedback.training.constructors import get_scheduler
from feedback.training.loop import train_epoch, validate_model
from feedback.datasets.std import EssayDs, get_targets
from transformers import AutoTokenizer
import pandas as pd

#def objective(trial, folddf, collate_fn):
if __name__ == '__main__':
    data = 'data/'
    params = {
        'epochs':12,
        'lr':0.000003,
        'min_lr': 0.004612,
        'warmup':174,
        'wd':0.007764,
        'b1':0.645282,
        'b2':0.784831,
        # 'batch_size':trial.suggest_int('batch_size',16,48),
    }
    params['batch_size'] = 16
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    collate_fn = DataCollatorWithPadding(tokenizer)
    train = pd.read_csv(f'{data}train.csv')
    targets = get_targets(train['discourse_effectiveness'])
    train_ds = EssayDs(train, tokenizer, targets)

    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], collate_fn=collate_fn, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small",num_labels=3)
#     model.cuda()
    opt = AdamW(model.parameters(),lr=params['lr'],weight_decay=params['wd'],betas=(params['b1'],params['b2']))
    sched = get_scheduler(params['warmup'], params['epochs']*len(train_dl),opt,params['lr']*params['min_lr'])
    vld = 0
#     for epoch in range(params['epochs']): 
#         loss = train_epoch(train_dl,model,opt,sched,torch.device('cuda'))
#         print('\033[1A', end='\x1b[2K')
#         print(f'Epoch: {epoch} loss: {loss}')

    torch.save(model.state_dict(), f'{data}full1.pt')
