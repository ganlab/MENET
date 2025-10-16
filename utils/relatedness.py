import torch
import pandas as pd
import numpy as np
from datetime import datetime

def Euclidean(feature, config):
    features_tensor = torch.tensor(feature.values, dtype=torch.float32)
    norms = features_tensor.norm(p=2, dim=1, keepdim=True)
    features_tensor = features_tensor / norms
    distance_matrix = torch.cdist(features_tensor, features_tensor, p=2)
    max_distance = torch.max(distance_matrix).item()
    gr = 1 - distance_matrix / max_distance
    gr_df = pd.DataFrame(gr.cpu().numpy(), index=feature.index, columns=feature.index)
    gr_df.index.name = "ID"
    gr_df.astype("float16")
    torch.save(gr_df, f"{config['gr_path']}/genetic_relatedness.pt", pickle_protocol=4)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Genetic Relatedness calculated successfully. "
          f"Saved at: {config['gr_path']}/genetic_relatedness.pt")

def calculate_genetic_relatedness(model, data, config):
    model.eval()
    with torch.no_grad():
        SNP = data.iloc[:, 1:].values
        SNP = torch.tensor(SNP, dtype=torch.float).reshape(SNP.shape[0], 1, SNP.shape[1])
        batch = 1024
        outputs = []
        for start in range(0, SNP.size(0), batch):
            end = min(start + batch, SNP.size(0))
            batch_inputs = SNP[start:end]
            output = model.forward_once(batch_inputs)
            outputs.append(output.detach().cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)
    output_df = pd.DataFrame(outputs, index=data.index)
    Euclidean(output_df, config)