"""Optimizer and learning-rate schedule factory."""
import torch


def build_optimizer(params, ocfg):
    if ocfg.name == "adam":
        opt = torch.optim.Adam(params, lr=ocfg.lr, betas=tuple(ocfg.betas), eps=ocfg.eps,
                               weight_decay=ocfg.weight_decay)
    elif ocfg.name == "adamw":
        opt = torch.optim.AdamW(params, lr=ocfg.lr, betas=tuple(ocfg.betas), eps=ocfg.eps,
                                weight_decay=ocfg.weight_decay)
    else:
        raise ValueError(ocfg.name)
    if ocfg.schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    elif ocfg.schedule == "exp_decay":        # reference code: lr0 * rate^(step/decay_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: ocfg.decay_rate ** (s / ocfg.decay_steps))
    else:
        raise ValueError(ocfg.schedule)
    return opt, sched
