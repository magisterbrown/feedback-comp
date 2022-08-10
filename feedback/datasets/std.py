from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

class EssayDs(Dataset):
    def __init__(self, df, tokinizer, targets, max_size=512):
        self.df = df
        self.tokinizer = tokinizer
        self.max_size = max_size
        self.targconv = targets
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['discourse_text']
        tokens = tokenizer(text,max_length=self.max_size) 
        
        target = self.targconv[self.df.iloc[idx]['discourse_effectiveness']]
        res = {
            'input_ids':tokens['input_ids'],
            'attention_mask':tokens['attention_mask'],
            'target':target
        } 
        return res

class FoldsEssays:
    def __init__(self,df,folds):
        self.df = df
        essays = df['essay_id'].unique()
        self.splits = np.array_split(essays, folds)
        
    def get_fold(self, test_id):
        
        test = self.df[self.df['essay_id'].isin(self.splits[test_id])]
        train = self.df[~self.df['essay_id'].isin(self.splits[test_id])]
        
        return train, test
    
class FoldsEssaysDs(FoldsEssays):
    def __init__(self,df, folds):
        super().__init__(df, folds)
        self.targets = get_targets(df['discourse_effectiveness'])
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    
    def get_fold(self, test_id):
        train, test = super().get_fold(test_id)
        train = EssayDs(train, self.tokenizer, self.targets)
        test = EssayDs(test, self.tokenizer, self.targets)
        
        return train, test
    
def get_targets(vals) -> dict:
    uni = vals.unique()
    res = dict()
    for key,v in enumerate(uni):
        res[v] = key
    return res
        
