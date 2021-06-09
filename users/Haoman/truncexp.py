import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """
    arg_constraints = {'rate': constraints.positive, 'upper': constraints.positive}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0


    def __init__(self, rate, validate_args=None):
        self.rate, = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        super(Exponential, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for ._exponential()
            u = torch.rand(shape, dtype=self.rate.dtype, device=self.rate.device)
            return -(-u).log1p() / self.rate
        return self.rate.new(shape).exponential_() / self.rate


      def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate *(self.upper - value)
