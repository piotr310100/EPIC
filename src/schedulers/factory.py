from .registry import schedulers


def create_scheduler(type, **kwargs):
    if type not in schedulers:
        raise ValueError(
            f"Unknown scheduler '{type}'. Available: {list(schedulers.keys())}"
        )
    return schedulers[type](**kwargs)
