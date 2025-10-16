import torch
import copy
from datetime import datetime
from sklearn.metrics import r2_score
import numpy as np
from utils.ig import ig_analysis

def setup_training_env(config, model, criterion):
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    return device, optimizer, scheduler

def forward_triplet_loss_step(model, criterion, batch, device):
    batch = [x.to(device) for x in batch]
    anchor, anchor_phen, pos, pos_phen, neg, neg_phen = batch
    anchor_out, pos_out, neg_out = model(anchor, pos, neg)
    return criterion(anchor_out, anchor_phen, pos_out, pos_phen, neg_out, neg_phen)

def train_trait_specific_encoder_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_train in train_loader:
        optimizer.zero_grad()
        loss = forward_triplet_loss_step(model, criterion, batch_train, device)
        loss.backward()
        optimizer.step()

def evaluate_trait_specific_encoder(model, dataloader, criterion, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_val in dataloader:
            loss = forward_triplet_loss_step(model, criterion, batch_val, device)
            val_loss += loss.item()
        val_loss /= len(dataloader)
    return val_loss


def train_trait_specific_encoder(config, model, train_loader, val_loader, criterion):
    device, optimizer, scheduler = setup_training_env(config, model, criterion)
    best_val_loss=float('inf')
    best_model = copy.deepcopy(model)
    for epoch in range(config['epoch']):
        train_trait_specific_encoder_one_epoch(model, train_loader, optimizer, criterion, device)
        train_loss = evaluate_trait_specific_encoder(model, train_loader, criterion, device)
        val_loss = evaluate_trait_specific_encoder(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"[Epoch {epoch + 1:03d}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Trait-specific encoder trained successfully. "
          f"Saved at: {config['model_path']}/trait_specific_encoder.pt")
    torch.save(best_model.state_dict(), f"{config['model_path']}/trait_specific_encoder.pt")
    return best_model


def train_menet_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x1, x2)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def evaluate_menet(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    y_preds, y_trues = [], []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_pred = model(x1, x2)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * x1.size(0)
            y_preds.extend(y_pred.detach().cpu().numpy())
            y_trues.extend(y.detach().cpu().numpy())
    y_preds = np.array(y_preds).flatten()
    y_trues = np.array(y_trues).flatten()
    avg_loss = total_loss / len(dataloader.dataset)
    r2 = r2_score(y_trues, y_preds)
    return avg_loss, r2, y_preds, y_trues

def train_menet(config, model, train_loader, val_loader, test_loader, criterion, tensor_for_ig, windows=None):
    device, optimizer, scheduler = setup_training_env(config, model, criterion)
    best_val_r2=float('-inf')
    best_model = copy.deepcopy(model)
    for epoch in range(config['epoch']):
        train_menet_one_epoch(model, train_loader, optimizer, criterion, device)
        train_loss, train_r2, _, _ = evaluate_menet(model, train_loader, criterion, device)
        val_loss, val_r2, _, _ = evaluate_menet(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        ig_ve, ig_repgeno = ig_analysis(model, tensor_for_ig[0], tensor_for_ig[1], device, windows)
        print(f"train_loss = {train_loss:.4f}, train_r2 = {train_r2:.4f}, "
              f"val_loss = {val_loss:.4f}, val_r2 = {val_r2:.4f}, "
              f"ig_VE = {ig_ve:.4f}, ig_RepGeno={ig_repgeno:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = copy.deepcopy(model)
    test_loss, test_r2, _, _ = evaluate_menet(best_model, test_loader, criterion, device)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Best model achieved R² = {test_r2:.4f} on the test set.")

if __name__ == '__main__':
    pass