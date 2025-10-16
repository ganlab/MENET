import torch
import numpy as np
import copy


def integrated_gradients_batch(model, emb_a, emb_b, baseline_a=None, baseline_b=None, steps=50, windows=None):
    device = emb_b.device
    B, D = emb_b.shape
    if baseline_a is None:
        baseline_a = torch.zeros_like(emb_a)

    if baseline_b is None:
        baseline_b = torch.zeros_like(emb_b)
    alphas_a = torch.linspace(0, 1, steps, device=device).view(-1, *([1] * (baseline_a.dim())))
    alphas_b = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1)
    interp_a = baseline_a.unsqueeze(0) + alphas_a * (emb_a.unsqueeze(0) - baseline_a.unsqueeze(0))
    interp_b = baseline_b.unsqueeze(0) + alphas_b * (emb_b.unsqueeze(0) - baseline_b.unsqueeze(0))
    if windows:
        interp_a_flat = interp_a.reshape(emb_a.shape[0], -1, D).detach().requires_grad_(True)
    else:
        interp_a_flat = interp_a.reshape(-1, D).detach().requires_grad_(True)

    interp_b_flat = interp_b.reshape(-1, D).detach().requires_grad_(True)
    output = model(interp_a_flat, interp_b_flat)
    total_output = output.sum()
    grads = torch.autograd.grad(
        outputs=total_output,
        inputs=[interp_a_flat, interp_b_flat],
        create_graph=False
    )

    grad_a, grad_b = grads[0], grads[1]

    if windows:
        grad_a = grad_a.view(steps, emb_a.shape[0], B, D)
    else:
        grad_a = grad_a.view(steps, B, D)

    grad_b = grad_b.view(steps, B, D)

    avg_grad_a = grad_a.mean(dim=0)
    avg_grad_b = grad_b.mean(dim=0)

    ig_a = (emb_a - baseline_a) * avg_grad_a
    ig_b = (emb_b - baseline_b) * avg_grad_b
    if windows:
        ig_a = ig_a.mean(dim=0)
    return ig_a, ig_b


def ig_analysis(model, VE, GR, device, windows=None):
    model.eval()
    ig_a_ls = []
    ig_b_ls = []
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    for start in range(0, GR.size(0), 64):
        end = min(start + 64, GR.size(0))
        ve_inputs = VE[start:end].to(device)
        gr_inputs = GR[start:end].to(device)

        if windows:
            ve_inputs = torch.split(ve_inputs, list(windows.values()), dim=-1)
            ve_embedding = torch.stack([model.ve[i](ve_inputs[i]) for i in range(len(windows))], dim=0)
        else:
            ve_embedding = model.ve(ve_inputs)
        repgeno_embedding = model.repGeno(gr_inputs)
        ig_a, ig_b = integrated_gradients_batch(model_copy.fusion, ve_embedding, repgeno_embedding, steps=50, windows=windows)
        ig_a_ls.append(ig_a.detach().cpu())
        ig_b_ls.append(ig_b.detach().cpu())
    ig_a_all = torch.cat(ig_a_ls, dim=0).numpy()
    ig_b_all = torch.cat(ig_b_ls, dim=0).numpy()
    importance_ve = np.abs(ig_a_all).mean()
    importance_repgeno = np.abs(ig_b_all).mean()
    total = importance_ve + importance_repgeno
    normalized_ve = importance_ve / total
    normalized_repgeno = importance_repgeno / total
    return normalized_ve, normalized_repgeno


if __name__ == '__main__':
    pass


