from .decay import ExponentialDecay, LinearDecay, LogDecay

schedulers = {
    "exp": ExponentialDecay,
    "lin": LinearDecay,
    "log": LogDecay,
}
