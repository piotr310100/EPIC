import math

from .base import SchedulerBase


class ExponentialDecay(SchedulerBase):
    def __init__(self, num_prototypes_start=100, num_prototypes_end=5, num_steps=10, **kwargs):
        super().__init__(num_prototypes_start, num_prototypes_end, num_steps, **kwargs)
        self.gamma = (num_prototypes_end / num_prototypes_start) ** (1 / num_steps)

    def get_value(self, step):
        return int(self.num_prototypes_start * (self.gamma**step))


class LinearDecay(SchedulerBase):
    def __init__(self, num_prototypes_start=100, num_prototypes_end=5, num_steps=10, **kwargs):
        super().__init__(num_prototypes_start, num_prototypes_end, num_steps, **kwargs)
        self.slope = (num_prototypes_start - num_prototypes_end) / num_steps

    def get_value(self, step):
        return int(max(self.num_prototypes_end, self.num_prototypes_start - self.slope * step))


class LogDecay(SchedulerBase):
    def __init__(self, num_prototypes_start=100, num_prototypes_end=5, num_steps=10, c=0.1, **kwargs):
        super().__init__(num_prototypes_start, num_prototypes_end, num_steps, **kwargs)
        self.c = c

    def get_value(self, step):
        return int(
            self.num_prototypes_end
            + (self.num_prototypes_start - self.num_prototypes_end)
            * (
                math.log(1 + self.c * (self.num_steps - step))
                / math.log(1 + self.c * self.num_steps)
            )
        )
