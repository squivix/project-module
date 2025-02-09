import math

from torch.optim.lr_scheduler import LRScheduler


class CosineLRScheduler(LRScheduler):
    def __init__(self, optimizer, max_lr=0.0001, min_lr=1e-6, n_warmup=5, m_end=1, max_epochs=50, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = n_warmup
        self.steady_end_steps = m_end
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1  # Current epoch
        if epoch <= self.warmup_steps:
            return [self.max_lr for _ in self.base_lrs]  # Keep it constant during warmup
        elif epoch > self.max_epochs - self.steady_end_steps:
            return [self.min_lr for _ in self.base_lrs]  # Keep it constant in the last m_end steps
        else:
            # Cosine Annealing Formula
            factor = (epoch - self.warmup_steps) / (self.max_epochs - self.warmup_steps - self.steady_end_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * factor))
            return [lr for _ in self.base_lrs]
