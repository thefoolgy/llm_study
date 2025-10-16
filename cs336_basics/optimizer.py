from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    # def step(self, closure: Optional[Callable] = None):
    #     loss = None if closure is None else closure()
    #     for group in self.param_groups:
    #         lr = group["lr"]
    #         betas = group["betas"]
    #         eps = group["eps"]
    #         weight_decay = group["weight_decay"]
    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #             state = self.state[p]
    #             if len(state) == 0:
    #                 state['step'] = 0
    #                 state['m'] = torch.zeros_like(p.data)
    #                 state['v'] = torch.zeros_like(p.data)
    #             state['step'] += 1
    #             t = state['step']
    #             m, v = state['m'], state['v'] 
    #             grad = p.grad.data
    #             beta1, beta2 = betas 
    #             m.mul_(beta1).add_(grad, alpha=1 - beta1)
    #             v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    #             bias_correction1 = 1 - beta1 ** t
    #             bias_correction2 = 1 - beta2 ** t
    #             step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    #             # p.data -= step_size * m / math.sqrt(v + eps)
    #             p.data.addcdiv_(m, (v.sqrt().add_(eps)), value=-step_size)
    #             p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
    #     return loss
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups: 
            lr = group["lr"] 
            betas = group["betas"] 
            eps = group["eps"] 
            weight_decay = group["weight_decay"] 
            for p in group["params"]: 
                if p.grad is None: 
                    continue 
                state = self.state[p] 
                # import pdb; pdb.set_trace()
                t = state.get("step", 0) + 1
                state['step'] = t
                m = state.get("m", torch.zeros_like(p)) 
                v = state.get("v", torch.zeros_like(p)) 
                # if len(state) == 0:
                #     state['step'] = 0
                #     state['m'] = torch.zeros_like(p.data)
                #     state['v'] = torch.zeros_like(p.data)
                # state['step'] += 1
                # t = state['step']
                # m, v = state['m'], state['v'] 
                grad = p.grad.data 
                m = betas[0] * m + (1 - betas[0]) * grad 
                v = betas[1] * v + (1- betas[1]) * grad**2 
                beta1, beta2 = betas 
                bias_correction1 = 1 - beta1 ** t 
                bias_correction2 = 1 - beta2 ** t 
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1 
                p.data -= step_size * m / torch.sqrt(v + eps) 
                p.data -= lr * weight_decay * p.data 
                state["m"] = m 
                state["v"] = v

        return loss