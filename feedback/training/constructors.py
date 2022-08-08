from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
def get_scheduler(warmup:int, steps:int, optimizer, min_lr=0):
    
    dec_steps = steps-warmup
    decay_sch = CosineAnnealingLR(optimizer, dec_steps, eta_min=min_lr)
    finsched = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup, after_scheduler=decay_sch)
    return finsched
