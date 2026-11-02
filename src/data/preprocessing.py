import os
import re
import pandas as pd
import numpy as np

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

def extract_competencies(row):
    """
    Extrai as 5 competências (C1 a C5) e converte para classes de 0 a 5 (divisões por 40).
    """
    grades = row.get('grades')
    if isinstance(grades, str):
        numbers = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', grades)]
    elif isinstance(grades, (list, np.ndarray)):
        numbers = [float(x) for x in grades]
    else:
        numbers = []

    # Se a lista contiver 6 valores [C1, C2, C3, C4, C5, Total]
    if len(numbers) >= 5:
        c1, c2, c3, c4, c5 = numbers[0:5]
    else:
        c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0

    # Converter notas (0, 40, 80, 120, 160, 200) para classes ordinais (0, 1, 2, 3, 4, 5)
    return pd.Series({
        'c1_class': int(np.clip(c1 // 40, 0, 5)),
        'c2_class': int(np.clip(c2 // 40, 0, 5)),
        'c3_class': int(np.clip(c3 // 40, 0, 5)),
        'c4_class': int(np.clip(c4 // 40, 0, 5)),
        'c5_class': int(np.clip(c5 // 40, 0, 5)),
        'c1_score': c1,
        'c2_score': c2,
        'c3_score': c3,
        'c4_score': c4,
        'c5_score': c5,
        'total_score': c1 + c2 + c3 + c4 + c5
    })

def process_aes_enem():
    print("Processando AES-ENEM exatamente conforme o artigo...")
    raw_dir = os.path.join(RAW_PATH, "aes_enem")
    out_dir = os.path.join(PROCESSED_PATH, "aes_enem")
    os.makedirs(out_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(raw_dir, "sourceAWithGraders_train.csv"))
    val_df = pd.read_csv(os.path.join(raw_dir, "sourceAWithGraders_validation.csv"))
    test_df = pd.read_csv(os.path.join(raw_dir, "sourceAWithGraders_test.csv"))
    
    for df in [train_df, val_df, test_df]:
        comp_df = df.apply(extract_competencies, axis=1)
        for col in comp_df.columns:
            df[col] = comp_df[col]

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    
    print("Dataset processado com sucesso!")

if __name__ == "__main__":
    process_aes_enem()