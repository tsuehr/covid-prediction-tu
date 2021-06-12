import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
x = torch.tensor(x)
alpha = 4
beta = 50
y = alpha * x + beta*x**2

alpha_op = torch.tensor(2., requires_grad=True)
beta_op = torch.tensor(4., requires_grad=True)

for i in range(10):
    pred = alpha_op * x + beta_op*x**2
    loss = (y - pred).pow(2).sum()
    loss += (((alpha - alpha_op)**4)*0.01).item()

    loss.backward()

    with torch.no_grad():
        alpha_op -= alpha_op.grad * 0.01
        beta_op -= beta_op.grad * 0.01
        print(loss, beta_op, beta_op.grad, alpha_op, alpha_op.grad)
        alpha_op.grad = None
        beta_op.grad = None

    plt.plot(pred.detach().numpy())
    plt.plot(y)
    plt.show()

'''
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


'''