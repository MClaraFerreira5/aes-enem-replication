import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

class AESDataset(Dataset):
    def __init__(self, data_path, target_col="c1_class", tokenizer_name="neuralmind/bert-base-portuguese-cased", max_length=512, exclude_col=None):
        # data_path pode ser um caminho ou uma lista de caminhos (concatenados)
        if isinstance(data_path, (list, tuple)):
            self.data = pd.concat([pd.read_parquet(p) for p in data_path], ignore_index=True)
        else:
            self.data = pd.read_parquet(data_path)
        # exclude_col: coluna booleana; linhas True são removidas (ex.: in_aes_enem)
        if exclude_col is not None:
            if exclude_col not in self.data.columns:
                raise ValueError(f"Coluna '{exclude_col}' não existe nos dados. Rode o pré-processamento novamente.")
            self.data = self.data[~self.data[exclude_col].astype(bool)].reset_index(drop=True)
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

def get_test_dataloader(processed_dir="data/processed/aes_enem", target_col="c1_class", split="test", batch_size=8, tokenizer_name="neuralmind/bert-base-portuguese-cased", exclude_col=None):
    """
    DataLoader apenas de avaliação. split="test" usa test.parquet;
    split="all" concatena train + val + test (útil para avaliar em um dataset inteiro).
    """
    if split == "all":
        paths = [f"{processed_dir}/{name}.parquet" for name in ("train", "val", "test")]
    else:
        paths = f"{processed_dir}/{split}.parquet"
    ds = AESDataset(paths, target_col=target_col, tokenizer_name=tokenizer_name, exclude_col=exclude_col)
    print(f"Avaliando em {len(ds)} redações ({split})")
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
