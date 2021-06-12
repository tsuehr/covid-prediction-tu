import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, truncexpon, truncnorm, nbinom, norm
import pandas as pd
import time
import torch
from torch import nn
from torch import distributions
from torch import rand
from torch import autograd
from torch import optim
import pickle

# %%

np.random.seed(seed=101)
torch.manual_seed(101)
torch.use_deterministic_algorithms(True)
dtype = torch.float64
device = torch.device("cpu")

# %%

data = pd.read_csv('../../data/covid19model.csv')

toy_data = pickle.load(open("toy.pickle", "rb"))


# %%

def trunc_exponential(scale, upper):
    sample = torch.distributions.exponential.Exponential(1 / scale).rsample()
    sample = sample / torch.tensor(1 - torch.exp(-upper / scale))
    return sample

def trunc_normal(mu, sigma, under, upper):
    distribution = torch.distributions.normal.Normal(loc=mu, scale=sigma, validate_args=None)
    normal_sample = distribution.rsample()
    cumulative = distribution.cdf(torch.tensor(upper)) - distribution.cdf(torch.tensor(under))
    return normal_sample / cumulative

def bij_transform(R0_prime, lower, upper):
    # Recieves a value in [-inf, inf] and returns value in [low, upper]
    bij = 1 / (1 + torch.exp(-R0_prime))
    scale = upper - lower
    return scale * bij + lower

def calc_prior_loss(tau, phi, R0, alpha, sigma):
    # log likelihood wrt. our prior ("regularisation")
    # ll stands for log-likelihood
    ll = torch.tensor(0.0, device=device)

    # dist_tau_t = distributions.exponential.Exponential(torch.tensor([1/0.03]))
    # ll += dist_tau_t.log_prob(tau).item()

    # dist_y = distributions.exponential.Exponential(tau) #the parameter in the brasket should either be float or tensor, to avoid any inconvienience,
    # I use everything as tensor. NOTE:tau_t is already a tensor.
    # ll += dist_y.log_prob(y).item()

    # dist_phi = distribution.normal.Normal(loc=torch.tensor([25]), scale=torch.tensor([10]))
    ll += dist_phi.log_prob(phi).item()

    # dist_R0 = distribution.normal.Normal(loc=torch.tensor([3.6]), scale=torch.tensor([0.8]))
    ll += dist_R0.log_prob(R0).item()

    # dist_alpha = distribution.normal.Normal(loc=torch.tensor([0.01]), scale=torch.tensor([0.01]))
    ll += dist_alpha.log_prob(alpha).item()

    # dist_sigma = distribution.normal.Normal(loc=torch.tensor([0.1]), scale=torch.tensor([0.3]))
    ll += dist_sigma.log_prob(sigma).item()

    return ll

def seed_init_infect(y):
    # Initialize newly_infected, cumulative_infected, St
    newly_infected = torch.zeros(num_observations, device=device, dtype=dtype)  # number of newly infected
    cumulative_infected = torch.zeros(num_observations, device=device)  # cumulative number of infected

    St = torch.zeros(num_observations, device=device)  # fraction of susceptible population
    # seed initial infection / impute first num_impute days
    newly_infected[0:num_impute] = y.clone()
    cumulative_infected[0] = 0.
    cumulative_infected[1:num_impute] = torch.cumsum(newly_infected[0:num_impute - 1].clone(), dim=0)
    St[0:num_impute] = torch.tensor([torch.maximum(population.clone() - x, torch.tensor(0)) / population for x in
                                     cumulative_infected[0:num_impute].clone()])
    return newly_infected, cumulative_infected, St

def calc_Rt(R0, epsilon_t, sigma, ll):
    # Initialize eta_t
    eta_t = torch.zeros(num_observations, device=device)  # transformed reproduction number

    # calculate Rt: the basic reproduction number
    # basic reproduction number as a latent random walk
    beta_0 = torch.log(R0)
    eta_t[0] = beta_0
    for t in range(1, num_observations):
        dist_epsilon_t = torch.distributions.Normal(epsilon_t[t - 1], sigma)
        ll += dist_epsilon_t.log_prob(epsilon_t[t])
    eta_t[1:num_observations] = beta_0 + epsilon_t[0:num_observations - 1].clone()
    Rt = torch.exp(eta_t)
    return Rt, ll

def calc_infections(cumulative_infected, newly_infected, St, Rt):
    # Initialize effectively_infectious
    effectively_infectious = torch.zeros(num_observations, device=device)  # effective number of infectious individuals

    # calculate infections
    for t in range(num_impute, num_observations):
        # Update cumulative newly_infected
        cumulative_infected[t] = cumulative_infected[t - 1].clone() + newly_infected[t - 1].clone()
        # Adjusts for portion of pop that are susceptible
        St[t] = torch.maximum(population.clone() - cumulative_infected[t].clone(), cero) / population.clone()
        # effective number of infectous individuals
        ni_temp = newly_infected[:t].view(1, 1, -1).clone()
        si_temp = torch.flip(serial_interval, (0,))[-t:].view(1, 1, -1)
        effectively_infectious[t] = torch.nn.functional.conv1d(ni_temp, si_temp)

        newly_infected[t] = St[t].clone() * Rt[t].clone() * effectively_infectious[t].clone()
    return newly_infected

def calc_hospit(newly_infected, alpha):
    # Initialize expected_daily_hospit
    expected_daily_hospit = torch.zeros(num_observations, device=device)  # expected number of daily hospitalizations

    # calculate expected number of hospitalizations
    expected_daily_hospit[0] = (1e-15) * newly_infected[0].clone()
    for t in range(1, num_observations):
        ni_temp = newly_infected[:t].view(1, 1, -1)
        pi_temp = torch.flip(pi, (0,))[-t - 1:-1].view(1, 1, -1)
        expected_daily_hospit[t] = torch.nn.functional.conv1d(ni_temp, pi_temp)
    expected_daily_hospit = alpha * expected_daily_hospit
    return expected_daily_hospit

def compare_results(expected_daily_hospit, phi, ll):
    # compare observed hospitalizations to model results
    # likelihood of the data wrt. to the model

    for i in range(0, num_observations):
        p = 1 / (1 + expected_daily_hospit[i] / phi)
        dist = torch.distributions.negative_binomial.NegativeBinomial(phi, p - torch.tensor(2.225e-5))
        ll += dist.log_prob(torch.round(observed_daily_hospit[i]))
    return ll

def forward_pass():
    # Initialize y
    y = trunc_exponential(tau, 1000)
    R0 = bij_transform(R0_prime, lower=2, upper=5)
    phi = bij_transform(phi_prime, lower=0, upper=50)
    alpha = bij_transform(alpha_prime, lower=0, upper=0.05)

    # Calculate prior loss
    ll = calc_prior_loss(tau, phi, R0, alpha, sigma)

    # Seed initial infections
    newly_infected, cumulative_infected, St = seed_init_infect(y)

    # Calculate Rt & random walk loss
    Rt, ll = calc_Rt(R0, epsilon_t, sigma, ll)

    # Calculate infections
    newly_infected = calc_infections(cumulative_infected, newly_infected, St, Rt)

    # Calculate expected hospitalizations
    expected_daily_hospit = calc_hospit(newly_infected, alpha)

    # Compare observed hospitalizations to model results
    ll = compare_results(expected_daily_hospit, phi, ll)
    return expected_daily_hospit, Rt, ll, R0, phi, alpha



# Initialization


cero = torch.tensor(0., requires_grad=False, device=device, dtype=dtype)
num_impute = 6
observed_daily_hospit = toy_data['observed_daily']
pi = torch.tensor(data.delay_distr, requires_grad=False, device=device, dtype=dtype)
serial_interval = torch.tensor(data.serial_interval, requires_grad=False, device=device, dtype=dtype)
population = torch.tensor(5793636, requires_grad=False, device=device, dtype=dtype)
num_observations = len(observed_daily_hospit)


## Initialize latent variables/parameters


tau = torch.tensor(np.random.exponential(1 / 0.03), requires_grad=True, device=device, dtype=dtype)
sigma = torch.tensor(truncnorm.rvs((0 - 0.1) / 0.3, (0.5 - 0.1) / 0.3, loc=0.1, scale=0.3), requires_grad=True,
                     device=device,
                     dtype=dtype)  # positive, tricky, gamma or inverse gamma, log normal  --> try something out, large sigma--> prone to overfitting # standart deviation of random walk step

phi_prime = torch.tensor(truncnorm.rvs((0 - 25) / 10, (np.inf - 25) / 10, loc=25, scale=10), requires_grad=True,
                   device=device,
                   dtype=dtype)  # has to be positive, between 0-50 --> uniform # dispersion (shape) parameter for observations
alpha_prime = torch.tensor(truncnorm.rvs((0 - 1 / 100) / 1 / 100, (5 / 100 - 1 / 100) / 1 / 100, loc=1 / 100, scale=1 / 100),
                     requires_grad=True, device=device,
                     dtype=dtype)  # uniform distribution between (0-5%) # probability to get hospitalized
R0_prime = torch.rand(1, dtype=dtype, device=device, requires_grad=True)

epsilon_t = torch.zeros(num_observations, device=device)
epsilon_t[0] = torch.distributions.Normal(cero, sigma.detach()).rsample()
for t in range(1, num_observations):
    epsilon_t[t] = torch.distributions.Normal(epsilon_t[t - 1].detach(), sigma.detach()).rsample()
epsilon_t.requires_grad_(True)

# %%

dist_tau_t = distributions.exponential.Exponential(torch.tensor([1 / 0.03], device=device))
# tau_t = dist_tau_t.sample()

dist_y = distributions.exponential.Exponential(tau)
# y = dist_y.sample()

dist_phi = distributions.normal.Normal(loc=torch.tensor([25], device=device), scale=torch.tensor([10], device=device))
# phi = dist_phi.sample()

dist_R0 = distributions.normal.Normal(loc=torch.tensor([3.6], device=device), scale=torch.tensor([0.8], device=device))
# R0 = dist_R0.sample()

dist_alpha = distributions.normal.Normal(loc=torch.tensor([0.01], device=device),
                                         scale=torch.tensor([0.01], device=device))
# alpha = dist_alpha.sample()

dist_sigma = distributions.normal.Normal(loc=torch.tensor([0.1], device=device),
                                         scale=torch.tensor([0.3], device=device))


# Optimization

# %%

learning_rate = 1e-12
epochs = 100
complete_time = time.time()

for k in range(epochs):
    start_time = time.time()
    decay = (1 - (k / (epochs * 1e5))) ** 2
    learning_rate = learning_rate * decay

    # forward pass - calculate expected_daily_hospit
    expected_daily_hospit, Rt, ll, R0, phi, alpha = forward_pass()

    # backward pass
    loss = -ll
    loss.backward()

    if k % 5 == 0:
        print(f'Time Step: {k}|| Loss: {loss},  R0":{R0_prime}, grad: {R0_prime.grad}, alpha": {alpha_prime} grad: {alpha_prime.grad}, phi": {phi_prime} grad: {phi_prime.grad}, sigma: {sigma} grad {sigma.grad}, epsilon_t.mean: {epsilon_t.mean()} grad.mean {epsilon_t.grad.mean()}')
        #print(f'Time Step: {k}||R0":{R0_prime}, grad: {R0_prime.grad}, R0:{R0}')

    with torch.no_grad():  # this part is SGD. can also replace with loss.step
        tau -= learning_rate * tau.grad
        phi_prime -= learning_rate * phi_prime.grad
        # R0_prime -= learning_rate * R0_prime.grad
        R0_prime -= 1e-4 * R0_prime.grad
        alpha_prime -= learning_rate * alpha_prime.grad
        sigma -= learning_rate * sigma.grad
        epsilon_t -= learning_rate * epsilon_t.grad * 1e+8

        tau.grad = None
        phi_prime.grad = None
        R0_prime.grad = None
        alpha_prime.grad = None
        sigma.grad = None
        epsilon_t.grad = None

    if k % 20 == 0:
        plt.plot(expected_daily_hospit.cpu().detach().numpy(), label='expected_daily_hospit')
        plt.plot(observed_daily_hospit.cpu().detach().numpy(), label='observed_daily_hospit')
        plt.legend()
        plt.show()


