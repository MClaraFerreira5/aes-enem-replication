import argparse
import json
import os
import shutil

import numpy as np
import pytorch_lightning as pl
import torch
from coral_pytorch.dataset import corn_label_from_logits
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.dataset import get_dataloaders, get_test_dataloader
from src.models.bertimbau_module import BERTimbauForAES
from src.utils.metrics import evaluate_predictions

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
COMPETENCIES = ["c1", "c2", "c3", "c4", "c5"]
DATASETS = {
    "aes_enem": "data/processed/aes_enem",
    "essay_br": "data/processed/essay_br",
}


def checkpoint_path(train_dataset, comp_name):
    """Local fixo onde o melhor checkpoint de cada competência fica salvo."""
    return os.path.join("checkpoints", train_dataset, comp_name, "best.ckpt")


def train_competence(
        comp_name,
        train_dataset="aes_enem",
        model_name=MODEL_NAME,
        batch_size=8,
        max_epochs=15,
        lr=3e-5,
        patience=2
    ):
    """Treina um modelo para uma competência e salva o melhor checkpoint em checkpoints/."""
    pl.seed_everything(42, workers=True)

    print(f"\n---> Treinando modelo para {comp_name.upper()} em {train_dataset} <---")

    dataloaders = get_dataloaders(
        processed_dir=DATASETS[train_dataset],
        target_col=f"{comp_name}_class",
        batch_size=batch_size,
        tokenizer_name=model_name
    )

    model = BERTimbauForAES(
        model_name=model_name,
        mode="ordinal",
        num_classes=6,  # 6 níveis: 0, 40, 80, 120, 160, 200
        lr=lr
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else 32,
        callbacks=[ckpt, es],
        enable_checkpointing=True
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])

    # Copia o melhor checkpoint para um caminho fixo (permite avaliar depois sem retreinar)
    dest = checkpoint_path(train_dataset, comp_name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(ckpt.best_model_path, dest)
    print(f"Melhor checkpoint salvo em {dest}")
    return dest


def predict_competence(
        comp_name,
        ckpt_file,
        test_dataset="aes_enem",
        test_split="test",
        model_name=MODEL_NAME,
        batch_size=8,
        exclude_overlap=False
    ):
    """Carrega um checkpoint e prediz as notas (0..200) no conjunto de teste escolhido."""
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {ckpt_file}. "
            "Treine primeiro (sem --skip-training) ou copie o checkpoint para esse caminho."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTimbauForAES.load_from_checkpoint(ckpt_file, map_location=device)
    model.eval()
    model.to(device)

    test_loader = get_test_dataloader(
        processed_dir=DATASETS[test_dataset],
        target_col=f"{comp_name}_class",
        split=test_split,
        batch_size=batch_size,
        tokenizer_name=model_name,
        exclude_col="in_aes_enem" if exclude_overlap else None
    )

    preds_scores = []
    real_scores = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            pred_classes = corn_label_from_logits(logits)

            # Converter classe (0..5) para nota real (0..200)
            preds_scores.extend((pred_classes * 40).cpu().numpy())
            real_scores.extend((labels * 40).cpu().numpy())

    return np.array(preds_scores), np.array(real_scores)


def save_results(results, path_prefix):
    """Salva os resultados em .json e em tabela markdown (mesmo estilo de baseline_results.md)."""
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [
        "| Alvo | QWK | RMSE | Discrepância (>80p) |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['target']} | {r['qwk']:.4f} | {r['rmse']:.2f} | {r['discrepancy_gt80']:.2f}% |")
    with open(f"{path_prefix}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResultados salvos em {path_prefix}.md e {path_prefix}.json")


def run_full_pipeline(
        train_dataset="aes_enem",
        test_dataset=None,
        test_split="test",
        competencies=None,
        skip_training=False,
        exclude_overlap=False,
        batch_size=8,
        max_epochs=15,
        lr=3e-5,
        patience=2
    ):
    test_dataset = test_dataset or train_dataset
    competencies = competencies or COMPETENCIES
    all_preds, all_reals, results = {}, {}, []

    print(f"Treino: {train_dataset} | Teste: {test_dataset} ({test_split}) | "
          f"Competências: {', '.join(competencies)}")

    for comp in competencies:
        if skip_training:
            ckpt_file = checkpoint_path(train_dataset, comp)
        else:
            ckpt_file = train_competence(
                comp, train_dataset=train_dataset, batch_size=batch_size,
                max_epochs=max_epochs, lr=lr, patience=patience
            )

        preds, reals = predict_competence(
            comp, ckpt_file, test_dataset=test_dataset, test_split=test_split,
            batch_size=batch_size, exclude_overlap=exclude_overlap
        )
        all_preds[comp] = preds
        all_reals[comp] = reals

        results.append(evaluate_predictions(reals, preds, target_name=f"Competência {comp.upper()}"))

    # Nota total (0 a 1000) = soma das 5 competências (só faz sentido com as 5)
    if set(competencies) == set(COMPETENCIES):
        total_preds = sum(all_preds[comp] for comp in COMPETENCIES)
        total_reals = sum(all_reals[comp] for comp in COMPETENCIES)

        print("\n==================================================")
        print(" RESULTADO FINAL: NOTA TOTAL DA REDAÇÃO (0 - 1000)")
        print("==================================================")
        results.append(evaluate_predictions(total_reals, total_preds, target_name="Nota Total ENEM"))

    suffix = "_sem_sobreposicao" if exclude_overlap else ""
    save_results(results, os.path.join("experiments", f"results_{train_dataset}_on_{test_dataset}{suffix}"))
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Treina/avalia o BERTimbau (CORN) para correção automática de redações do ENEM."
    )
    p.add_argument("--train-dataset", choices=list(DATASETS), default="aes_enem",
                   help="Dataset usado para treinar (ou de onde vêm os checkpoints se --skip-training).")
    p.add_argument("--test-dataset", choices=list(DATASETS), default=None,
                   help="Dataset de avaliação. Padrão: o mesmo do treino.")
    p.add_argument("--test-split", choices=["test", "all"], default="test",
                   help="'test' usa só test.parquet; 'all' usa o dataset inteiro (train+val+test).")
    p.add_argument("--skip-training", action="store_true",
                   help="Não treina: só carrega os checkpoints em checkpoints/<train-dataset>/ e avalia.")
    p.add_argument("--exclude-overlap", action="store_true",
                   help="Remove do teste (essay_br) as redações que também existem no AES-ENEM.")
    p.add_argument("--competencies", nargs="+", choices=COMPETENCIES, default=COMPETENCIES)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--patience", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        test_split=args.test_split,
        competencies=args.competencies,
        skip_training=args.skip_training,
        exclude_overlap=args.exclude_overlap,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience=args.patience,
    )
