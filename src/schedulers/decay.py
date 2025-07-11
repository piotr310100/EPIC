import math

from .base import SchedulerBase


class ExponentialDecay(SchedulerBase):
    def __init__(self, initial_value=100, final_value=5, num_steps=10, **kwargs):
        super().__init__(initial_value, final_value, num_steps, **kwargs)
        self.gamma = (final_value / initial_value) ** (1 / num_steps)

    def get_value(self, step):
        return int(self.initial_value * (self.gamma**step))


class LinearDecay(SchedulerBase):
    def __init__(self, initial_value=100, final_value=5, num_steps=10, **kwargs):
        super().__init__(initial_value, final_value, num_steps, **kwargs)
        self.slope = (initial_value - final_value) / num_steps

    def get_value(self, step):
        return int(max(self.final_value, self.initial_value - self.slope * step))


class LogDecay(SchedulerBase):
    def __init__(self, initial_value=100, final_value=5, num_steps=10, c=0.1, **kwargs):
        super().__init__(initial_value, final_value, num_steps, **kwargs)
        self.c = c

    def get_value(self, step):
        return int(
            self.final_value
            + (self.initial_value - self.final_value)
            * (
                math.log(1 + self.c * (self.num_steps - step))
                / math.log(1 + self.c * self.num_steps)
            )
        )
