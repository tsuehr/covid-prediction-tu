import numpy as np
import torch
import matplotlib.pyplot as plt

'''x = np.linspace(0, 1, 100)
x = torch.tensor(x)
alpha = 4
y = alpha * x
tau = torch.tensor(10., requires_grad=True)

for i in range(10):
    pred = tau * x
    loss = (y - pred).pow(2).sum()
    loss.backward()

    with torch.no_grad():
        tau -= tau.grad * 0.01
        tau.grad = None
    plt.plot(pred.detach().numpy())
    plt.plot(y)
    plt.show()'''


def trunc_exponential(scale, upper=1000):

    random_sample = torch.distributions.exponential.Exponential(1/scale).rsample()
    random_sample = random_sample/torch.tensor(1-torch.exp(-upper/scale))

    return random_sample

dtype = torch.float64
device = torch.device("cpu")
x = np.linspace(0, 1, 100)
x = torch.tensor(x)
alpha = 40
beta = 2
y = alpha * x + beta

tau_t = torch.tensor(1., requires_grad=True, device=device, dtype=dtype)
c = torch.tensor(10., requires_grad=True, device=device, dtype=dtype)

for i in range(1000):
    print(tau_t, c)
    action = trunc_exponential(tau_t, upper=1000)
    #loss = -m.log_prob(action)

    pred = action * x + c
    loss = (y - pred).pow(2).sum()
    loss.backward()

    with torch.no_grad():
        c -= c.grad*0.001
        tau_t -= tau_t.grad*0.001

        tau_t.grad = None
        c.grad = None

