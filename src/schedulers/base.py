class SchedulerBase:
    def __init__(self, num_prototypes_start=100, num_prototypes_end=5, num_steps=10, **kwargs):
        self.num_prototypes_start = num_prototypes_start
        self.num_prototypes_end = num_prototypes_end
        self.num_steps = num_steps

    def get_value(self, step):
        raise NotImplementedError("Subclasses must implement this method.")

    def __iter__(self):
        for step in range(self.num_steps):
            yield self.get_value(step)
