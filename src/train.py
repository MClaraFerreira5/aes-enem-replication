import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from coral_pytorch.dataset import corn_label_from_logits

from src.data.dataset import get_dataloaders
from src.models.bertimbau_module import BERTimbauForAES
from src.utils.metrics import evaluate_predictions

def train_and_eval_competence(
        comp_name, 
        model_name="neuralmind/bert-base-portuguese-cased", 
        batch_size=8, 
        max_epochs=2,
        lr=3e-5,
        patience=2
    ):
    print(f"\n---> Treinando Modelo para {comp_name.upper()} <---")
    
    dataloaders = get_dataloaders(
        processed_dir="data/processed/aes_enem",
        target_col=f"{comp_name}_class",
        batch_size=batch_size,
        tokenizer_name=model_name
    )

    model = BERTimbauForAES(
        model_name=model_name,
        mode="ordinal",
        num_classes=6,  # 6 níveis: 0, 40, 80, 120, 160, 200
        lr=2e-5
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else 32,
        enable_checkpointing=False
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])

    # Predição no conjunto de teste
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds_scores = []
    real_scores = []

    with torch.no_grad():
        for batch in dataloaders['test']:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            pred_classes = corn_label_from_logits(logits)
            
            # Converter classe (0..5) para nota real (0..200)
            preds_scores.extend((pred_classes * 40).cpu().numpy())
            real_scores.extend((labels * 40).cpu().numpy())

    return np.array(preds_scores), np.array(real_scores)

def run_full_pipeline():
    competencies = ['c1', 'c2', 'c3', 'c4', 'c5']
    all_preds = {}
    all_reals = {}

    for comp in competencies:
        preds, reals = train_and_eval_competence(comp)
        all_preds[comp] = preds
        all_reals[comp] = reals
        
        evaluate_predictions(reals, preds, target_name=f"Competência {comp.upper()}")

    # Somar a predição das 5 competências para obter a Nota Total (0 a 1000)
    total_preds = sum(all_preds[comp] for comp in competencies)
    total_reals = sum(all_reals[comp] for comp in competencies)

    print("\n==================================================")
    print(" RESULTADO FINAL: NOTA TOTAL DA REDAÇÃO (0 - 1000)")
    print("==================================================")
    evaluate_predictions(total_reals, total_preds, target_name="Nota Total ENEM")

if __name__ == "__main__":
    run_full_pipeline()