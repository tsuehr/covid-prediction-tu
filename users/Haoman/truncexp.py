import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class TruncatedExponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """
    arg_constraints = {'rate': constraints.positive, 'upper': constraints.real}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0


    def __init__(self, rate, upper, validate_args=None):
        self.rate, self.upper= broadcast_all(rate, upper)
        if isinstance(rate, Number) and isinstance(upper, Number):
            batch_shape = torch.Size()    
        else: 
            self.rate.size()           
        super(TruncatedExponential, self).__init__(batch_shape, validate_args=validate_args)


      def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate *(self.upper - value)
