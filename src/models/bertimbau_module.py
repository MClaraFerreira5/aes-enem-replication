import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

class BERTimbauForAES(pl.LightningModule):
    """
    Módulo PyTorch Lightning para Avaliação Automática de Redações (AES)
    suportando regressão contínua, classificação multiclasse e regressão ordinal (CORN).
    """
    def __init__(
        self, 
        model_name="neuralmind/bert-base-portuguese-cased", 
        mode="regression",       
        num_classes=6,           # Exemplo: 6 classes de nota para cada competência (0, 40, 80, 120, 160, 200 -> 0 a 5)
        lr=2e-5,
        weight_decay=0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Carregar backbone do BERTimbau
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(0.2)
        
        # Ajustar a cabeça de saída de acordo com o modo
        if self.mode == "regression":
            self.classifier = nn.Linear(self.config.hidden_size, 1)
            self.loss_fn = nn.MSELoss()
        elif self.mode == "multiclass":
            self.classifier = nn.Linear(self.config.hidden_size, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.mode == "ordinal":
            # Para CORN loss, a saída deve ter (num_classes - 1) neurônios
            self.classifier = nn.Linear(self.config.hidden_size, num_classes - 1)
            # CORN loss é chamada diretamente no training_step
        else:
            raise ValueError("Modo inválido! Escolha entre 'regression', 'multiclass' ou 'ordinal'.")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Utilizar a representação do token [CLS] (pooler_output ou primeira posição)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self.forward(input_ids, attention_mask)

        if self.mode == "regression":
            loss = self.loss_fn(logits.squeeze(-1), labels)
        elif self.mode == "multiclass":
            loss = self.loss_fn(logits, labels.long())
        elif self.mode == "ordinal":
            loss = corn_loss(logits, labels.long(), num_classes=self.num_classes)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self.forward(input_ids, attention_mask)

        if self.mode == "regression":
            loss = self.loss_fn(logits.squeeze(-1), labels)
            preds = logits.squeeze(-1)
        elif self.mode == "multiclass":
            loss = self.loss_fn(logits, labels.long())
            preds = torch.argmax(logits, dim=1)
        elif self.mode == "ordinal":
            loss = corn_loss(logits, labels.long(), num_classes=self.num_classes)
            preds = corn_label_from_logits(logits)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer