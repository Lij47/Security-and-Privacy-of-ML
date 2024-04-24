import torch
import torch.nn.functional as F
import lightning as L
import torch.nn as nn
import torchmetrics as tm
from audiodiffusion import AudioDiffusion
from audiodiffusion.mel import Mel

class InpaintAttack(L.LightningModule):
    def __init__(self, model_config, optimizer_config, lr_configs):
        super().__init__()
        self.save_hyperparameters()

        self.audio_diffusion = AudioDiffusion(model_id="teticio/audio-diffusion-breaks-256")
        self.mel = Mel()

        audio_length = 130560
        audio_length_sec = 5
        gap_size = 0.01
        self.window_size = int(round(audio_length/audio_length_sec * gap_size * 256 / audio_length))
        self.num_gaps = 10
        self.num_inpaints = 1
        self.audio_length = audio_length

        #shape of audio is 130560
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5120, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        self.optimizer_configs = optimizer_config
        self.lr_configs = lr_configs
        self.valid_accuracy = tm.Accuracy(task="binary")
        self.valid_precision = tm.Precision(task="binary")
        self.valid_recall = tm.Recall(task="binary")
        self.valid_f1 = tm.F1Score(task="binary")
        self.test_accuracy = tm.Accuracy(task="binary")
        self.test_precision = tm.Precision(task="binary")
        self.test_recall = tm.Recall(task="binary")
        self.test_f1 = tm.F1Score(task="binary")

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        pred = pred.sigmoid()
        pred = pred.round()

        metrics = {
            "valid_acc": self.valid_accuracy(pred, target), 
            "valid_prec": self.valid_precision(pred, target),
            "valid_recall": self.valid_recall(pred, target), 
            "valid_f1": self.valid_f1(pred, target),
        }
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        pred = pred.sigmoid()
        pred = pred.round()

        metrics = {
            "test_acc": self.valid_accuracy(pred, target), 
            "test_prec": self.valid_precision(pred, target),
            "test_recall": self.valid_recall(pred, target), 
            "test_f1": self.valid_f1(pred, target),
        }
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_configs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.lr_configs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
        },
    }