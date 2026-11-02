import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

class AESDataset(Dataset):
    def __init__(self, data_path, target_col="c1_class", tokenizer_name="neuralmind/bert-base-portuguese-cased", max_length=512):
        self.data = pd.read_parquet(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['essay_text'])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(row[self.target_col]), dtype=torch.long)
        }
        return item

def get_dataloaders(processed_dir="data/processed/aes_enem", target_col="c1_class", batch_size=8, tokenizer_name="neuralmind/bert-base-portuguese-cased"):
    train_ds = AESDataset(f"{processed_dir}/train.parquet", target_col=target_col, tokenizer_name=tokenizer_name)
    val_ds = AESDataset(f"{processed_dir}/val.parquet", target_col=target_col, tokenizer_name=tokenizer_name)
    test_ds = AESDataset(f"{processed_dir}/test.parquet", target_col=target_col, tokenizer_name=tokenizer_name)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    }