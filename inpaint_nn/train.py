import os
import torch
import torch.utils.data as du
from torch.utils.data import random_split
import lightning as L
import yaml
from inpaint_nn.model import InpaintAttack
from inpaint_nn.dataset import AudioData
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics as tm


def main(yaml_file_dir="configs/config.yaml"):

    torch.manual_seed(42)

    #load config file
    cfg = yaml.safe_load(open(yaml_file_dir))

    # load dataset 70/10/20 split
    print("Loading dataset...")
    train_dataset = AudioData(cfg["train_dataset"])
    valid_dataset = AudioData(cfg["valid_dataset"])
    test_dataset = AudioData(cfg["test_dataset"])
    print("Finished loading dataset.")

    # split the dataset into a train/validation/test set 
    print("Splitting dataset...")
    print("Finished splitting dataset.")

    print("Creating dataloaders...")
    train_loader = du.DataLoader(train_dataset, **cfg["train_dataloader"])
    valid_loader = du.DataLoader(valid_dataset, **cfg["valid_dataloader"])
    test_loader = du.DataLoader(test_dataset, **cfg["test_dataloader"])
    print("Finished creating dataloaders.")

    # load model
    print("Loading model...")
    model = InpaintAttack(cfg["model"], cfg["optimizer"], cfg["lr_scheduler"])
    print("Finished loading model.")

    checkpoint_callback = ModelCheckpoint(**cfg["callbacks"]["model_checkpoint"])
    earlystopping_callback = EarlyStopping(**cfg["callbacks"]["early_stopping"])
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_callback]

    # train model
    if cfg["logger"] is not None:
        logger = TensorBoardLogger(**cfg["logger"])
        trainer = L.Trainer(**cfg["trainer"], logger=logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(**cfg["trainer"], callbacks=[checkpoint_callback])


    print("Training model...")

    trainer.fit(model, train_loader, valid_loader, ckpt_path=cfg["last_checkpoint"])
    print("Finished training model.")

    print("Testing model...")
    trainer.test(model, test_loader)
    print("Finished testing model.")

def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--config', dest='config', default='inpaint_nn/configs/config.yaml')
    parser.add_argument('--saveckpt', dest='saveckpt', default='training/outputs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
    






