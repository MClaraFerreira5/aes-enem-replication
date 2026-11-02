import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

class AESDataset(Dataset):
    """
    Dataset PyTorch para carregar e tokenizar redações do AES-ENEM.
    """
    def __init__(self, data_path, tokenizer_name="neuralmind/bert-base-portuguese-cased", max_length=512, target_col="score"):
        self.data = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
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
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

        # Extração dos rótulos (scores)
        if self.target_col in row:
            item['labels'] = torch.tensor(row[self.target_col], dtype=torch.float)

        return item

def get_dataloaders(processed_dir="data/processed/aes_enem", batch_size=8, tokenizer_name="neuralmind/bert-base-portuguese-cased", target_col="score"):
    """
    Cria os DataLoaders para treino, validação, test_easy e test_hard.
    """
    train_ds = AESDataset(f"{processed_dir}/train.parquet", tokenizer_name=tokenizer_name, target_col=target_col)
    val_ds = AESDataset(f"{processed_dir}/val.parquet", tokenizer_name=tokenizer_name, target_col=target_col)
    test_easy_ds = AESDataset(f"{processed_dir}/test_easy.parquet", tokenizer_name=tokenizer_name, target_col=target_col)
    test_hard_ds = AESDataset(f"{processed_dir}/test_hard.parquet", tokenizer_name=tokenizer_name, target_col=target_col)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test_easy': DataLoader(test_easy_ds, batch_size=batch_size, shuffle=False),
        'test_hard': DataLoader(test_hard_ds, batch_size=batch_size, shuffle=False)
    }

    return dataloaders