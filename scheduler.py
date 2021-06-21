import math
from torch.optim.lr_scheduler import _LRScheduler


class CircularLRBeta:
    """ A learning rate updater that implements the CircularLearningRate scheme.
        Learning rate is increased then decreased linearly.
        Args:
            optimizer(torch.optim.optimizer) : None 
            lr_max(float) : the highest LR in the schedule.
            lr_divider(int) : Determined first iterarion LR. It starts from lr_max/lr_divider.
            cut_point(int) : None
            step_size(int) : during how many epochs will the LR go up from the lower bound, up to the upper bound.
            momentum(list) : Optional; momentum
    """
    def __init__(
        self, optimizer, lr_max, lr_divider, cut_point, step_size, momentum=None
    ):
        self.lr_max = lr_max
        self.lr_divider = lr_divider
        self.cut_point = step_size // cut_point
        self.step_size = step_size
        self.iteration = 0
        self.cycle_step = int(step_size * (1 - cut_point / 100) / 2)
        self.momentum = momentum
        self.optimizer = optimizer

    def get_lr(self):
        if self.iteration > 2 * self.cycle_step:
            cut = (self.iteration - 2 * self.cycle_step) / (
                self.step_size - 2 * self.cycle_step
            )
            lr = self.lr_max * (1 + (cut * (1 - 100) / 100)) / self.lr_divider

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        else:
            cut = self.iteration / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        return lr

    def get_momentum(self):
        if self.iteration > 2 * self.cycle_step:
            momentum = self.momentum[0]

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        else:
            cut = self.iteration / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        return momentum

    def step(self):
        lr = self.get_lr()

        if self.momentum is not None:
            momentum = self.get_momentum()

        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                group['betas'] = (momentum, group['betas'][1])

        return lr
    

class CosineAnnealingWithWarmupAndHardRestart(_LRScheduler):
    """ 
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps(int): Linear warmup step size. 
        cycle_steps (int): Cycle step size.
        max_lr(float): First cycle's max learning rate.
        min_lr(float): Min learning rate.
    """
    def __init__(
        self, optimizer, warmup_steps, cycle_steps, max_lr, min_lr=None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr / 50

        super(CosineAnnealingWithWarmupAndHardRestart, self).__init__(optimizer=optimizer)
        
        self.init_lr()
        
    def init_lr(self):
        self.lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.lrs.append(self.min_lr)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return (
                self.min_lr + 
                (self.max_lr - self.min_lr) / self.warmup_steps * self._step_count
            )
        else:
            x = (self._step_count - self.warmup_steps) % self.cycle_steps
            return (
                self.min_lr + 
                0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi / self.cycle_steps * x))
            )

    def step(self):
        self.lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self._step_count += 1

        
class CosineDecayWithWarmup(_LRScheduler):
    """ 
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps(int): Linear warmup step size.
        total_steps (int): Total step size.
        max_lr(float): First cycle's max learning rate.
        min_lr(float): Min learning rate.
    """
    def __init__(
        self, optimizer, warmup_steps, total_steps, max_lr, min_lr=None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr / 50

        super(CosineDecayWithWarmup, self).__init__(optimizer=optimizer)
        
        self.init_lr()
        
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return (
                self.min_lr + 
                (self.max_lr - self.min_lr) / self.warmup_steps * self._step_count
            )
        else:
            x = self._step_count - self.warmup_steps
            return (
                self.min_lr + 
                (self.max_lr - self.min_lr) / 2 * 
                (1 + math.cos((self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps) * math.pi))
            )

    def step(self):
        self.lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self._step_count += 1
        