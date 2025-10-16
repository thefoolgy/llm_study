import math
import torch
import numpy as np

def cosine_annealing_scheduler(step, lr_max, lr_min, step_w, step_c):
    #warm-up
    # if step < step_w:
    #     lr = (step / step_w) * lr_max
    # elif step_w <= step and step <= step_c:
    #     lr = lr_min + 0.5 * (1 + math.cos((step-step_w)*math.pi/(step_c - step_w))) * (lr_max - lr_min)
    # elif step > step_c:
    #     lr = lr_min
    # return lr
    step = min(step, step_c)

    if step < step_w:
        # Linear warm-up from 0 → lr_max
        lr = (step / step_w) * lr_max
    else:
        # Cosine annealing from lr_max → lr_min
        progress = (step - step_w) / (step_c - step_w)
        lr = lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)

    return lr

def gradient_clipping(params, max_l2_norm, eps = 1e-6):
    # grads = [p.grad for p in params if p.grad is not None]
    # if not grads:
    #     return 
    # total_norm = torch.sqrt(sum(torch.sum(g.detach()**2) for g in grads))
    # if total_norm > max_l2_norm:
    #     scale = max_l2_norm / (total_norm + eps)
    #     for g in grads:
    #         g.mul_(scale)
    flattened = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])
    l2_norm = torch.norm(flattened, p=2)
    if l2_norm < max_l2_norm:
        pass
    else:
        clip_coef = max_l2_norm / (l2_norm + 1e-6) 
        # parameters.div_(max_l2_norm/(l2_norm + torch.finfo(torch.float32).eps))
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    n = len(x)
    ix = np.random.randint(0, n-context_length, size = batch_size)
    x_batch = np.stack([x[i:i+context_length] for i in ix])
    y_batch = np.stack([x[i+1:i+context_length+1] for i in ix])

    x_batch = torch.tensor(x_batch, dtype = torch.long, device = device)
    y_batch = torch.tensor(y_batch, dtype = torch.long, device = device)

    return x_batch, y_batch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration



