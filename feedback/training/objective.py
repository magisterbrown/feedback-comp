import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from feedback.training.constructors import get_scheduler
from feedback.training.loop import train_epoch, validate_model

def objective(trial, folddf, collate_fn):
    
    params = {
        'epochs':trial.suggest_int('epochs',8,15),
        'lr':trial.suggest_float('lr',1e-6,1e-5,log=True),
        'min_lr':trial.suggest_float('min_lr',1e-3,1e-1,log=True),
        'warmup':trial.suggest_int('warmup',50,1000,log=True),
        'wd':trial.suggest_float('wd',1e-6,1e-2,log=True),
        'b1':trial.suggest_float('b1',0.5,0.95),
        'b2':trial.suggest_float('b2',0.5,0.9999,log=True),
        # 'batch_size':trial.suggest_int('batch_size',16,48),
    }
    params['batch_size'] = 16
    train_ds, test_ds = folddf.get_fold(0)

    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], collate_fn=collate_fn, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    
    test_dl = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn, 
                              num_workers=2, shuffle=False, pin_memory=True, drop_last=False)
    
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small",num_labels=3)
    model.cuda()
    opt = AdamW(model.parameters(),lr=params['lr'],weight_decay=params['wd'],betas=(params['b1'],params['b2']))
    sched = get_scheduler(params['warmup'], params['epochs']*len(train_dl),opt,params['lr']*params['min_lr'])
    vld = 0
    for epoch in range(params['epochs']): 
        loss = train_epoch(train_dl,model,opt,sched,torch.device('cuda'))
        vld = validate_model(model,test_dl,torch.device('cuda'))
        trial.report(vld, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
      
    return vld