class SchedulerBase:
    def __init__(self, initial_value=100, final_value=5, num_steps=10, **kwargs):
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps

    def get_value(self, step):
        raise NotImplementedError("Subclasses must implement this method.")

    def __iter__(self):
        for step in range(self.num_steps):
            yield self.get_value(step)
