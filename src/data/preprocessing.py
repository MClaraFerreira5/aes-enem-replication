import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

def split_easy_hard(df, threshold=80):
    """
    Divide o conjunto de teste em 'easy' e 'hard' com base na discrepância
    entre os avaliadores (conforme feito no artigo do AES-ENEM).
    """
    # Exemplo: se houver a coluna de variância/discrepância das notas entre corretores
    if 'grader_discrepancy' in df.columns:
        df_easy = df[df['grader_discrepancy'] <= threshold].copy()
        df_hard = df[df['grader_discrepancy'] > threshold].copy()
    else:
        # Se os dados originais já tiverem flags de concordância ou cálculo equivalente
        df_easy = df.copy()
        df_hard = df.copy()
        
    return df_easy, df_hard

def process_aes_enem():
    print("Processando AES-ENEM...")
    raw_dir = os.path.join(RAW_PATH, "aes_enem")
    out_dir = os.path.join(PROCESSED_PATH, "aes_enem")
    os.makedirs(out_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(raw_dir, "sourceAWithGraders_train.csv"))
    val_df = pd.read_csv(os.path.join(raw_dir, "sourceAWithGraders_validation.csv"))
    test_df = pd.read_csv(os.path.join(raw_dir, "sourceAWithGraders_test.csv"))
    sourceB_df = pd.read_csv(os.path.join(raw_dir, "sourceB_full.csv"))
    
    # Criar as divisões test_easy e test_hard
    test_easy, test_hard = split_easy_hard(test_df)
    
    # Salvar em Parquet para leitura ultra-rápida no PyTorch
    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_easy.to_parquet(os.path.join(out_dir, "test_easy.parquet"), index=False)
    test_hard.to_parquet(os.path.join(out_dir, "test_hard.parquet"), index=False)
    sourceB_df.to_parquet(os.path.join(out_dir, "sourceB.parquet"), index=False)
    
    print("AES-ENEM processado e salvo em data/processed/aes_enem/!")

def process_essay_br():
    print("Processando Essay-Br...")
    raw_file = os.path.join(RAW_PATH, "essay_br", "extended_essay-br.csv")
    out_dir = os.path.join(PROCESSED_PATH, "essay_br")
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(raw_file)
    # Limpeza básica ou mapeamento de colunas se necessário
    df.to_parquet(os.path.join(out_dir, "clean_essay_br.parquet"), index=False)
    print("Essay-Br processado e salvo em data/processed/essay_br/!")

if __name__ == "__main__":
    process_aes_enem()
    process_essay_br()