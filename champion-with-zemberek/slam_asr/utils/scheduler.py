"""Learning rate schedulers for SLAM-ASR."""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_constant_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int = 1000,
) -> LambdaLR:
    """Create a scheduler that linearly increases the learning rate during warmup,

    then keeps it constant at the maximum learning rate (Paper Section 3.3).
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int = 1000,
    num_training_steps: int = 100000,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a scheduler that linearly increases during warmup and then decays with cosine curve."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return LambdaLR(optimizer, lr_lambda)
