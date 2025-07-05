import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import pandas as pd
from utils.process_data.utils import job_Euclidean
from sklearn.decomposition import PCA
from utils.process_model.ig_analysis import ig_analysis


def repgeno_backbone_train(model: nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           criterion: nn.Module,
                           optimizer: torch.optim.Optimizer,
                           num_epochs: int,
                           device: str,
                           data:pd.DataFrame,
                           scheduler: Optional[optim.lr_scheduler._LRScheduler],
                           ):
    model.to(device)
    for epoch in range(num_epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss = 0
        n_train = 0
        for batch_train in train_loader:
            x_train, y_train = batch_train
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            train_loss += loss.item() * x_train.size(0)
            n_train += x_train.size(0)


        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_val in val_loader:
                x_val, y_val = batch_val
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred = model(x_val)
                loss = criterion(y_pred, y_val)
                val_loss += loss.item() * x_val.shape[0]
                n_val += x_val.shape[0]

        print(f"epoch = {epoch}, train_loss = {train_loss / n_train}, val_loss = {val_loss / n_val}", end="\n")

        scheduler.step(train_loss / n_train)

        if epoch in [num_epochs]:
            feature = data.iloc[:, 1:]
            inputs = torch.tensor(feature.values, dtype=torch.float32).to(device)
            model.eval()
            batch = 1024
            outputs = []
            with torch.no_grad():
                for start in range(0, inputs.size(0), batch):
                    end = min(start + batch, inputs.size(0))
                    batch_inputs = inputs[start:end]
                    output = model.embedding(batch_inputs)
                    outputs.append(output.detach().cpu().numpy())
            outputs = np.concatenate(outputs, axis=0)
            output_df = pd.DataFrame(outputs, index=feature.index)
            file_path = f"saved/genetic_relatedness.csv"
            job_Euclidean(output_df, file_path)
    return model


def menet_train(model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                test_dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                num_epochs: int,
                device: str,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                ggr
                ):
    GR = ggr[1]
    VE = ggr[0]
    VE = torch.tensor(VE.values, dtype=torch.float).to(device)
    VE = VE.reshape(VE.shape[0], 1, -1)
    GR = torch.tensor(GR.values, dtype=torch.float).to(device)
    GR = GR.reshape(GR.shape[0], 1, -1)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred1, y_pred2, y_pred3 = model(x1, x2)
            loss = criterion(y_pred1, y) + criterion(y_pred2, y) + criterion(y_pred3, y)
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss = 0
        n_train = 0
        with torch.no_grad():
            for batch_train in train_dataloader:
                x1_train, x2_train, y_train = batch_train
                x1_train, x2_train, y_train = x1_train.to(device), x2_train.to(device), y_train.to(device)
                y_pred1, y_pred2, y_pred3 = model(x1_train, x2_train)
                loss = criterion(y_pred1, y_train) + criterion(y_pred2, y_train) + criterion(y_pred3, y_train)
                train_loss += loss.item() * x1_train.shape[0]
                n_train += x1_train.shape[0]

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_val in val_dataloader:
                x1_val, x2_val, y_val = batch_val
                x1_val, x2_val, y_val = x1_val.to(device), x2_val.to(device), y_val.to(device)
                y_pred1, y_pred2, y_pred3 = model(x1_val, x2_val)
                loss = criterion(y_pred1, y_val) + criterion(y_pred2, y_val) + criterion(y_pred3, y_val)
                val_loss += loss.item() * x1_val.shape[0]
                n_val += x1_val.shape[0]

        model.eval()
        with torch.no_grad():
            for batch_test in test_dataloader:
                pass
        scheduler.step(val_loss / n_val)

        ig_ve, ig_repgeno = ig_analysis(model, VE, GR)
        print(f"epoch = {epoch}, train_loss = {train_loss / n_train:.4f}, val_loss = {val_loss / n_val:.4f},",
              f"ig_ve: {ig_ve:.4f}, ig_repgeno: {ig_repgeno:.4f}")
    return model