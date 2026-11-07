import argparse
import ast
import os
import re

import numpy as np
import pandas as pd

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

COMPETENCIES = ["c1", "c2", "c3", "c4", "c5"]


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


# ---------------------------------------------------------------------------
# EssayBR
# ---------------------------------------------------------------------------

def essay_to_text(value):
    """
    Converte o campo 'essay' em texto corrido.
    Aceita string simples, lista/array de parágrafos ou a string de uma lista
    (formato do CSV bruto: "['parágrafo 1', 'parágrafo 2']").
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return "\n".join(str(p) for p in value)
    text = str(value)
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return "\n".join(str(p) for p in parsed)
        except (ValueError, SyntaxError):
            pass
    return text


def build_essay_br_frame(df):
    """
    Converte o formato do EssayBR (essay, c1..c5, score) para o formato que o
    restante do código espera (mesmas colunas do AES-ENEM processado):
    essay_text, c{n}_score, c{n}_class (0..5) e total_score.
    """
    missing = [c for c in ["essay", "score"] + COMPETENCIES if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no EssayBR: {missing}")

    out = df.copy()
    out["essay_text"] = out["essay"].map(essay_to_text)
    for c in COMPETENCIES:
        out[f"{c}_score"] = out[c].astype(float)
        out[f"{c}_class"] = np.clip(out[c] // 40, 0, 5).astype(int)
    out["total_score"] = sum(out[f"{c}_score"] for c in COMPETENCIES)
    return out


def _shingles(text, k=8):
    """Conjunto de hashes de janelas de k palavras consecutivas (para detectar textos repetidos)."""
    words = re.findall(r"\w+", str(text).lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(len(words) - k + 1, 0))}


def flag_overlap_with_aes_enem(df, text_col="essay_text", threshold=0.3):
    """
    Marca (coluna booleana in_aes_enem) as redações do EssayBR que já aparecem no AES-ENEM
    (sourceA: train + validation + test). Os dois datasets compartilham muitas redações;
    avaliar nelas um modelo treinado no AES-ENEM infla o resultado.
    Uma redação conta como repetida se mais de `threshold` (30%) das suas janelas de 8 palavras
    existem no AES-ENEM.
    """
    raw_dir = os.path.join(RAW_PATH, "aes_enem")
    aes_shingles = set()
    for split in ("train", "validation", "test"):
        path = os.path.join(raw_dir, f"sourceAWithGraders_{split}.csv")
        if not os.path.exists(path):
            print(f"  Aviso: {path} não encontrado; a coluna in_aes_enem não será criada.")
            return df
        for text in pd.read_csv(path)["essay_text"]:
            aes_shingles |= _shingles(text)

    def overlap(text):
        sh = _shingles(text)
        return bool(sh) and len(sh & aes_shingles) / len(sh) > threshold

    out = df.copy()
    out["in_aes_enem"] = out[text_col].map(overlap)
    print(f"  {int(out['in_aes_enem'].sum())} de {len(out)} redações do EssayBR também estão no AES-ENEM")
    return out


def split_dataframe(df, val_frac=0.1, test_frac=0.1, seed=42):
    """Divide aleatoriamente (e de forma reprodutível) em treino/validação/teste."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = shuffled.iloc[:n_test]
    val = shuffled.iloc[n_test:n_test + n_val]
    train = shuffled.iloc[n_test + n_val:]
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def process_essay_br(input_path=None, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Lê data/processed/essay_br/clean_essay_br.parquet e gera
    data/processed/essay_br/{train,val,test}.parquet (80/10/10 por padrão).
    """
    out_dir = os.path.join(PROCESSED_PATH, "essay_br")
    input_path = input_path or os.path.join(out_dir, "clean_essay_br.parquet")
    print(f"Processando EssayBR a partir de {input_path} ...")

    df = build_essay_br_frame(pd.read_parquet(input_path))
    df = flag_overlap_with_aes_enem(df)
    train_df, val_df, test_df = split_dataframe(df, val_frac, test_frac, seed)

    os.makedirs(out_dir, exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
        print(f"  {name}: {len(part)} redações")

    print("EssayBR processado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processamento dos datasets")
    parser.add_argument("--dataset", choices=["aes_enem", "essay_br", "all"],
                        default="aes_enem",
                        help="Qual dataset processar (padrão: aes_enem)")
    args = parser.parse_args()

    if args.dataset in ("aes_enem", "all"):
        process_aes_enem()
    if args.dataset in ("essay_br", "all"):
        process_essay_br()
