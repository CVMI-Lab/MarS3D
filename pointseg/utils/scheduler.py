import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(self, optimizer, milestones, steps_per_epoch, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(optimizer=optimizer,
                         milestones=[epoch * steps_per_epoch for epoch in milestones],
                         gamma=gamma,
                         last_epoch=last_epoch,
                         verbose=verbose)


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, milestones, steps_per_epoch, gamma=0.1,
                 warmup_epochs=5, warmup_ratio=1e-6, last_epoch=-1, verbose=False):
        milestones = [epoch * steps_per_epoch for epoch in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_epochs * steps_per_epoch:
                warmup_coefficient = 1 - (1 - s / warmup_epochs / steps_per_epoch) * (1 - warmup_ratio)
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor
        super().__init__(optimizer=optimizer,
                         lr_lambda=multi_step_with_warmup,
                         last_epoch=last_epoch,
                         verbose=verbose)


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, epochs, steps_per_epoch, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(optimizer=optimizer,
                         T_max=epochs * steps_per_epoch,
                         eta_min=eta_min,
                         last_epoch=last_epoch,
                         verbose=verbose)


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """
    def __init__(self,
                 optimizer,
                 max_lr,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=False,
                 last_epoch=-1,
                 verbose=False):
        super().__init__(optimizer=optimizer,
                         max_lr=max_lr,
                         epochs=epochs,
                         steps_per_epoch=steps_per_epoch,
                         pct_start=pct_start,
                         anneal_strategy=anneal_strategy,
                         cycle_momentum=cycle_momentum,
                         base_momentum=base_momentum,
                         max_momentum=max_momentum,
                         div_factor=div_factor,
                         final_div_factor=final_div_factor,
                         three_phase=three_phase,
                         last_epoch=last_epoch,
                         verbose=verbose)


def build_scheduler(cfg, optimizer):
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
