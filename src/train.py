import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.dataset import get_dataloaders
from src.models.bertimbau_module import BERTimbauForAES
from src.utils.metrics import evaluate_predictions
from coral_pytorch.dataset import corn_label_from_logits

def run_experiment(
    mode="ordinal",
    model_name="neuralmind/bert-base-portuguese-cased",
    target_col="score",
    num_classes=6,
    batch_size=8,
    max_epochs=3,
    lr=2e-5
):
    print(f"\n==================================================")
    print(f" Iniciando Treino: {model_name} | Modo: {mode}")
    print(f"==================================================\n")

    # 1. Carregar DataLoaders
    dataloaders = get_dataloaders(
        processed_dir="data/processed/aes_enem",
        batch_size=batch_size,
        tokenizer_name=model_name,
        target_col=target_col
    )

    # 2. Inicializar Modelo Lightning
    model = BERTimbauForAES(
        model_name=model_name,
        mode=mode,
        num_classes=num_classes,
        lr=lr
    )

    # 3. Configurar Callbacks do Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="data/outputs/checkpoints/",
        filename="bertimbau-baseline-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min"
    )

    # 4. Configurar Trainer (GPU/CPU automática)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else 32,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # 5. Executar Treinamento
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

    # 6. Avaliação nos splits de Teste (test_easy e test_hard)
    print("\nEvaluating on test_easy and test_hard...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = {}
    for split_name in ['test_easy', 'test_hard']:
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloaders[split_name]:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)

                if mode == "regression":
                    preds = logits.squeeze(-1)
                elif mode == "multiclass":
                    preds = torch.argmax(logits, dim=1)
                elif mode == "ordinal":
                    preds = corn_label_from_logits(logits)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        res = evaluate_predictions(all_labels, all_preds, target_name=f"Baseline ({split_name})")
        results[split_name] = res

    return results

if __name__ == "__main__":
    run_experiment()