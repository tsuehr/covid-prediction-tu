# %%

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
import torch.nn as nn

def bij_transform(prime, lower, upper):
    # Recieves a value in [-inf, inf] and returns value in [low, upper]
    bij = 1 / (1 + torch.exp(-prime / upper))
    scale = upper - lower
    return scale * bij + lower


def bij_transform_inv(transf, lower, upper):
    return -torch.log(((upper - lower) / (transf - lower) - 1) ** upper)


def bij_transform_inf(prime):
    return torch.exp(prime)

# %%

np.random.seed(seed=101)
torch.manual_seed(101)
torch.use_deterministic_algorithms(True)
dtype = torch.float64
device = torch.device("cpu")

# %%

data = pd.read_csv('data/covid19model.csv')

# %%

toy_data = pd.read_csv('data/toy_data_2.csv')


# %%

toy_data_np = toy_data['0'].round()
plt.plot(toy_data_np)

# %% md

# Toy Data
#tau = torch.tensor(33)  # bij_transform(tau_prime, lower=0, upper=200)
#y = torch.distributions.exponential.Exponential(1 / tau).rsample()
#R0 = torch.tensor(3.6)  # bij_transform(R0_prime, lower=2, upper=5)
#phi = torch.tensor(25)  # bij_transform(phi_prime, lower=1, upper=50)
#alpha = torch.tensor(0.01)  # bij_transform(alpha_prime, lower=0, upper=0.05)
#sigma = torch.tensor(0.1)  # bij_transform(sigma_prime, lower=0.0001, upper=0.5)

# %% md

# Initialization

# %%

cero = torch.tensor(0., requires_grad=False, device=device, dtype=dtype)
num_impute = 6
# observed_daily_hospit = torch.tensor(toy_data_np, requires_grad=False, device=device, dtype=dtype)
observed_daily_hospit = torch.tensor(data.hospit, requires_grad=False, device=device, dtype=dtype)
pi = torch.tensor(data.delay_distr, requires_grad=False, device=device, dtype=dtype)
serial_interval = torch.tensor(data.serial_interval, requires_grad=False, device=device, dtype=dtype)
population = torch.tensor(5793636, requires_grad=False, device=device, dtype=dtype)
num_observations = len(observed_daily_hospit)

# %% md

## Initialize latent variables/parameters

# %%

"""tau_prime = torch.tensor(np.random.exponential(1 / 0.03), requires_grad=True, device=device, dtype=dtype)
phi_prime = torch.tensor(truncnorm.rvs((0 - 25) / 10, (np.inf - 25) / 10, loc=25, scale=10), requires_grad=True,
                         device=device,
                         dtype=dtype)  # has to be positive, between 0-50 --> uniform # dispersion (shape) parameter for observations
R0_prime = torch.tensor(truncnorm.rvs((2 - 3.6) / 0.8, (5 - 3.6) / 0.8, loc=3.6, scale=0.8), requires_grad=True,
                        device=device,
                        dtype=dtype)  # probably gamma or inverse gamma distribution (compare to truncated normal) # initial reproduction number
alpha_prime = torch.tensor(
    truncnorm.rvs((0 - 1 / 100) / 1 / 100, (5 / 100 - 1 / 100) / 1 / 100, loc=1 / 100, scale=1 / 100),
    requires_grad=True, device=device,
    dtype=dtype)  # uniform distribution between (0-5%) # probability to get hospitalized
sigma_prime = torch.tensor(truncnorm.rvs((0 - 0.1) / 0.3, (0.5 - 0.1) / 0.3, loc=0.1, scale=0.3), requires_grad=True,
                           device=device,
                           dtype=dtype)  # positive, tricky, gamma or inverse gamma, log normal  --> try something out, large sigma--> prone to overfitting # standart deviation of random walk step"""

# %%
'''
tau_prime = torch.tensor(-200.0, requires_grad=True, device=device, dtype=dtype)
R0_prime = torch.tensor(-5.0, requires_grad=True, device=device, dtype=dtype)
phi_prime = torch.tensor(-5.0, requires_grad=True, device=device, dtype=dtype)
alpha_prime = torch.tensor(0.05, requires_grad=True, device=device, dtype=dtype)
sigma_prime = torch.tensor(0.05, requires_grad=True, device=device, dtype=dtype)
'''

tau_prime = torch.tensor(bij_transform_inv(torch.tensor(33), 0, 100), requires_grad=True, device=device, dtype=dtype)  # 33
R0_prime = torch.tensor(bij_transform_inv(torch.tensor(3.6), 2, 5), requires_grad=True, device=device, dtype=dtype)   # 3.6
phi_prime = torch.tensor(bij_transform_inv(torch.tensor(25), 0, 50), requires_grad=True, device=device, dtype=dtype)  # 25
alpha_prime = torch.tensor(bij_transform_inv(torch.tensor(0.01), 0.00001, 0.05), requires_grad=True, device=device, dtype=dtype)  # 0.01
sigma_prime = torch.tensor(bij_transform_inv(torch.tensor(0.1), 0.00001, 0.5), requires_grad=True, device=device, dtype=dtype)  # 0.1
sigma = bij_transform(sigma_prime, 0.00001, 0.5)

# %%

epsilon_t = torch.zeros(num_observations, device=device)
epsilon_t[0] = torch.distributions.Normal(cero, sigma.detach()).rsample()
for t in range(1, num_observations):
    epsilon_t[t] = torch.distributions.Normal(epsilon_t[t - 1].detach(), sigma.detach()).rsample()
epsilon_t.requires_grad_(True)

# %% md

## Init Distributions

# %%

dist_tau = distributions.exponential.Exponential(torch.tensor([1 / 0.03], device=device))

dist_phi = distributions.normal.Normal(loc=torch.tensor([25], device=device), scale=torch.tensor([10], device=device))

dist_R0 = distributions.normal.Normal(loc=torch.tensor([3.6], device=device), scale=torch.tensor([0.8], device=device))

dist_alpha = distributions.normal.Normal(loc=torch.tensor([0.01], device=device),
                                         scale=torch.tensor([0.01], device=device))

dist_sigma = distributions.normal.Normal(loc=torch.tensor([0.1], device=device),
                                         scale=torch.tensor([0.3], device=device))


# %% md

# Define Forward Pass

# %%




# %%

def calc_prior_loss(tau, phi, R0, alpha, sigma):
    # log likelihood wrt. our prior ("regularisation")
    # ll stands for log-likelihood
    ll = torch.tensor(0.0, device=device)

    ll += dist_tau.log_prob(tau)[0]  # TODO

    ll += dist_phi.log_prob(phi)[0]

    ll += dist_R0.log_prob(R0)[0]

    ll += dist_alpha.log_prob(alpha)[0]

    ll += dist_sigma.log_prob(sigma)[0]

    return -ll




# %%

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


# %%

def calc_Rt(R0, epsilon_t, sigma, ll):
    # Initialize eta_t
    eta_t = torch.zeros(num_observations, device=device)  # transformed reproduction number
    # calculate Rt: the basic reproduction number
    # basic reproduction number as a latent random walk
    beta_0 = torch.log(R0)
    eta_t[0] = beta_0

    # for t in range(1, num_observations):
    #    dist_epsilon_t = torch.distributions.Normal(epsilon_t[t - 1], sigma)
    #    ll += dist_epsilon_t.log_prob(epsilon_t[t])

    loc = epsilon_t[:-1].clone()
    scale = sigma * torch.ones(num_observations - 1)
    mvn = distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=torch.diag(scale))
    ll += mvn.log_prob(epsilon_t[1:].clone())

    eta_t[1:num_observations] = beta_0 + epsilon_t[0:num_observations - 1].clone()
    Rt = torch.exp(eta_t)
    ll = (-1) * ll
    return Rt, ll


# %%

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


# %%

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


# %%

def compare_results(expected_daily_hospit, phi, ll):
    # compare observed hospitalizations to model results
    # likelihood of the data wrt. to the model

    for i in range(0, num_observations):
        p = 1 / (1 + expected_daily_hospit[i] / phi)
        if p == 1:
            p = p.clone() - torch.tensor(2.225e-5)
        if p < 2.225e-5:
            p = p.clone() + torch.tensor(2.225e-5)
        dist = torch.distributions.negative_binomial.NegativeBinomial(phi, p - torch.tensor(2.225e-5))
        ll += dist.log_prob(observed_daily_hospit[i])

    ll = (-1) * ll
    return ll

def compare_results_abs(expected_daily_hospit, phi, ll):
    # compare observed hospitalizations to model results
    # likelihood of the data wrt. to the model

    diff = expected_daily_hospit - observed_daily_hospit
    square = diff.square()
    msr = square.mean()
    ll += msr


    return ll


# %%

def forward_pass():
    # Initialize y
    tau = bij_transform(tau_prime, 0, 100)
    y = torch.distributions.exponential.Exponential(1 / tau).rsample()
    R0 = bij_transform(R0_prime, lower=2, upper=5)
    phi = bij_transform(phi_prime, lower=0.00001, upper=50)
    alpha = bij_transform(alpha_prime, lower=0.00001, upper=0.05)
    sigma = bij_transform(sigma_prime, lower=0.00001, upper=0.5)

    # Calculate prior loss
    ll_prior = calc_prior_loss(tau, phi, R0, alpha, sigma)

    # Seed initial infections
    newly_infected, cumulative_infected, St = seed_init_infect(y)

    # Calculate Rt & random walk loss
    Rt, ll_rw = calc_Rt(R0, epsilon_t, sigma, torch.tensor(0.0, device=device))  ##ll

    # Calculate infections
    newly_infected = calc_infections(cumulative_infected, newly_infected, St, Rt)

    # Calculate expected hospitalizations
    expected_daily_hospit = calc_hospit(newly_infected, alpha)
    #x_np = expected_daily_hospit.detach().numpy()
    #x_df = pd.DataFrame(x_np)
    #x_df.to_csv('toy_data_2.csv')

    # Compare observed hospitalizations to model results
    ll_comp = compare_results_abs(expected_daily_hospit, phi, torch.tensor(0.0, device=device))  ##ll

    return expected_daily_hospit, Rt, ll_prior, ll_rw, ll_comp, tau, R0, phi, alpha, sigma


# %% md

# Optimization

# %%

# Visualization initialization
alpha_vis = []
sigma_vis = []
R0_vis = []
tau_vis = []
phi_vis = []
epsilon_mean_vis = []
loss_vis = []
prior_loss_vis = []
rw_loss_vis = []
comp_loss_vis = []
learning_rate_vis = []

learning_rate = 1e-2
epochs = 200
complete_time = time.time()

var_list = [tau_prime, phi_prime, R0_prime, alpha_prime, sigma_prime, epsilon_t]
optimizer = optim.Adam(var_list, lr=learning_rate)

for k in range(epochs):
    optimizer.zero_grad()
    start_time = time.time()
    decay = (1 - (k / (epochs * 1000))) ** 2
    learning_rate = learning_rate * decay

    # forward pass - calculate expected_daily_hospit
    expected_daily_hospit, Rt, ll_prior, ll_rw, ll_comp, tau, R0, phi, alpha, sigma = forward_pass()

    # backward pass
    loss = ll_comp  #+ ll_rw + ll_prior

    loss.backward()


    if k % 10 == 0:
        print(
            f'\nPrior Loss:{ll_prior}  Random Walk Loss:{ll_rw} Comparison Loss:{ll_comp} \n'
            f'\n\nTime Step: {k} || Loss: {loss} || Learning Rate: {learning_rate}\n\nR0:{R0}  grad:{R0_prime.grad}\nalpha:{alpha}  grad:{alpha_prime.grad}\n'
            f'phi:{phi}  grad:{phi_prime.grad}\nsigma:{sigma}  grad:{sigma_prime.grad}'
            f'\nepsilon_t.mean:{epsilon_t.mean()}  grad.mean:{epsilon_t.grad.mean()}\ntau:{tau}  grad:{tau_prime.grad}\n')
        print("This Run:  %s seconds" % (time.time() - start_time))
    optimizer.step()
    '''with torch.no_grad():  # this part is SGD. can also replace with loss.step
        tau_prime -= learning_rate * tau_prime.grad
        phi_prime -= learning_rate * phi_prime.grad
        R0_prime -= learning_rate * R0_prime.grad
        alpha_prime -= learning_rate * alpha_prime.grad
        sigma_prime -= learning_rate * sigma_prime.grad
        epsilon_t -= learning_rate * epsilon_t.grad

        tau_prime.grad = None
        phi_prime.grad = None
        R0_prime.grad = None
        alpha_prime.grad = None
        sigma_prime.grad = None
        epsilon_t.grad = None'''

    # Visualization
    alpha_vis.append(alpha)
    sigma_vis.append(sigma)
    R0_vis.append(R0)
    tau_vis.append(tau)
    phi_vis.append(phi)
    epsilon_mean_vis.append(epsilon_t.abs().mean())
    loss_vis.append(loss)
    prior_loss_vis.append(ll_prior)
    rw_loss_vis.append(ll_rw)
    comp_loss_vis.append(ll_comp)
    learning_rate_vis.append(learning_rate)

    if k % 10 == 0:
        plt.plot(expected_daily_hospit.cpu().detach().numpy(), label='expected_daily_hospit')
        plt.plot(observed_daily_hospit.cpu().detach().numpy(), label='observed_daily_hospit')
        plt.legend()
        plt.show()

    if k % 500 == 0:
        fig, axs = plt.subplots(4)
        fig.suptitle(f'Time step {k}')
        axs[0].plot(alpha_vis)
        axs[0].title.set_text('Alpha')
        axs[1].plot(sigma_vis)
        axs[1].title.set_text('Sigma')
        axs[2].plot(R0_vis)
        axs[2].title.set_text('R0')
        axs[3].plot(tau_vis)
        axs[3].title.set_text('Tau')
        fig.tight_layout()
        plt.show()

print("Complete Run:  %s seconds" % (time.time() - complete_time))

# %% md

# Evaluate Model

# %%

plt.plot(expected_daily_hospit.cpu().detach().numpy(), label='expected_daily_hospit')
plt.plot(observed_daily_hospit.cpu().detach().numpy(), label='observed_daily_hospit')
plt.legend()
plt.show()

# %%

fig, axs = plt.subplots(6)
fig.suptitle(f'Evaluate Values')
fig.set_figheight(10)
axs[0].plot(alpha_vis)
axs[0].title.set_text('Alpha')
axs[1].plot(sigma_vis)
axs[1].title.set_text('Sigma')
axs[2].plot(R0_vis)
axs[2].title.set_text('R0')
axs[3].plot(tau_vis)
axs[3].title.set_text('Tau')
axs[4].plot(phi_vis)
axs[4].title.set_text('Phi')
axs[5].plot(epsilon_mean_vis)
axs[5].title.set_text('Epsilon_Abs_Mean')
plt.show()

# %%

fig, axs = plt.subplots(5)
fig.suptitle(f'Loss Evaluation')
fig.set_figheight(10)
axs[0].plot(loss_vis)
axs[0].title.set_text('Total Loss')
axs[1].plot(prior_loss_vis)
axs[1].title.set_text('Prior Loss')
axs[2].plot(rw_loss_vis)
axs[2].title.set_text('Random Walk Loss')
axs[3].plot(comp_loss_vis)
axs[3].title.set_text('Comparison Loss')
axs[4].plot(learning_rate_vis)
axs[4].title.set_text('Learning Rate')
plt.show()
