from tqdm import tqdm
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, truncexpon, truncnorm, nbinom
import pandas as pd
import torch
from torch import nn
from torch import distributions

#%%

np.random.seed(seed=101)
torch.manual_seed(101)
torch.use_deterministic_algorithms(True)
dtype = torch.float64
device = torch.device("cpu")


#%%

data = pd.read_csv('../../data/covid19model.csv')
data_fake = pd.read_csv('expected_daily_hosp.csv', index_col=0)

#%%



#%%


def trunc_exponential(scale, upper):
    sample = torch.distributions.exponential.Exponential(1/scale).rsample()
    sample = sample/torch.tensor(1-torch.exp(-upper/scale))
    return sample
# torch.distributions.exponential.Exponential(1/scale).sample()/torch.tensor(1-torch.exp(-upper/scale))









learning_rate = 1e-5

for k in range(400):

# forward pass - calculate expected_daily_hospit
    loss = 0.
    cero = torch.tensor(0., requires_grad=False, device=device, dtype=dtype)
    num_impute = 6
    observed_daily_hospit = torch.tensor(data.hospit, requires_grad=False, device=device, dtype=dtype)
    pi = torch.tensor(data.delay_distr, requires_grad=False, device=device, dtype=dtype)
    serial_interval = torch.tensor(data.serial_interval, requires_grad=False, device=device, dtype=dtype)
    population = torch.tensor(5793636, requires_grad=False, device=device, dtype=dtype)
    num_observations = len(observed_daily_hospit)



    tau = torch.tensor(np.random.exponential(1 / 0.03), requires_grad=True, device=device, dtype=dtype)
    #tau = torch.tensor(8.384203233308204, requires_grad=True, device=device, dtype=dtype)
    # b=(upper-lower)/scale, loc=lower, scale=scale
    y = trunc_exponential(tau, 1000)  # number of initial newly_infected (seed)

    # For trunc ((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    phi = torch.tensor(truncnorm.rvs((0 - 25) / 10, (np.inf - 25) / 10, loc=25, scale=10), requires_grad=True,
                       device=device, dtype=dtype)  # dispersion (shape) parameter for observations
    #phi = torch.tensor(13.471380423795104, requires_grad=True,
    #                       device=device, dtype=dtype)  # dispersion (shape) parameter for observations
    R0 = torch.tensor(truncnorm.rvs((2 - 3.6) / 0.8, (5 - 3.6) / 0.8, loc=3.6, scale=0.8), requires_grad=True,
                      device=device, dtype=dtype)  # initial reproduction number
    #R0 = torch.tensor(2.1091230713336286, requires_grad=True,
    #                  device=device, dtype=dtype)
    alpha = torch.tensor(
        truncnorm.rvs((0 - 1 / 100) / 1 / 100, (5 / 100 - 1 / 100) / 1 / 100, loc=1 / 100, scale=1 / 100),
        requires_grad=True, device=device, dtype=dtype)  # probability to get hospitalized

    #alpha = torch.tensor(0.010002413299271573,
    #        requires_grad=True, device=device, dtype=dtype)  # probability to get hospitalized
    sigma = torch.tensor(truncnorm.rvs((0 - 0.1) / 0.3, (0.5 - 0.1) / 0.3, loc=0.1, scale=0.3), requires_grad=True,
                         device=device, dtype=dtype)  # standart deviation of random walk step

    #sigma = torch.tensor(0.209933952628366, requires_grad=True,
    #                     device=device, dtype=dtype)  # standart deviation of random walk step

    # log li elihood wrt. our prior ("regularisation")
    # ll stands for log-likelihood
    ll = torch.tensor(0.0)
    # Initialize time series variables
    newly_infected = torch.zeros(num_observations)  # number of newly infected
    effectively_infectious = torch.zeros(num_observations)  # effective number of infectious individuals
    expected_daily_hospit = torch.zeros(num_observations)  # expected number of daily hospitalizations
    cumulative_infected = torch.zeros(num_observations)  # cumulative number of infected
    eta_t = torch.zeros(num_observations)  # transformed reproduction number
    epsilon_t = torch.zeros(num_observations)  # random walk
    St = torch.zeros(num_observations)  # fraction of susceptible population



    '''
    ll += torch.tensor(expon.logpdf(tau_t, 1 / 0.03))
    ll += torch.tensor(truncexpon.logpdf(y,b=(1000 - 0) / tau_t, loc=0, scale=tau_t))
    ll += torch.tensor(truncnorm.logpdf(phi,(0 - 25) / 10, (np.inf - 25) / 10, loc=25, scale=10))
    ll += torch.tensor(truncnorm.logpdf(R0,(2 - 3.6) / 0.8, (5 - 3.6) / 0.8, loc=3.6, scale=0.8))
    ll += torch.tensor(truncnorm.logpdf(alpha,(0 - 1/100) / 1/100, (5/100 - 1/100) / 1/100, loc=1/100, scale=1/100))
    ll += torch.tensor(truncnorm.logpdf(sigma,(0 - 0.1) / 0.3, (0.5 - 0.1) / 0.3, loc=0.1, scale=0.3))
    '''

    # seed initial infection / impute first num_impute days
    newly_infected[0:num_impute] = y.clone()
    cumulative_infected[0] = 0.
    cumulative_infected[1:num_impute] = torch.cumsum(newly_infected[0:num_impute - 1].clone(), dim=0)
    St[0:num_impute] = torch.tensor([torch.maximum(population.clone() - x, torch.tensor(0)) / population for x in cumulative_infected[0:num_impute].clone()])

    # calculate Rt: the basic reproduction number
    # basic reproduction number as a latent random walk
    beta_0 = torch.log(R0)
    #epsilon_t[0] = torch.distributions.Normal(cero, sigma).rsample()
    #for t in range(1, num_observations):
    #    epsilon_t[t] = torch.distributions.Normal(epsilon_t[t - 1].clone(), sigma).rsample()
    #eta_t = beta_0 + epsilon_t  # + RNN[X_t, t]  # .clone() necessary?
    eta_t[0] = beta_0
    epsilon_t[0] = torch.distributions.Normal(cero, sigma).rsample()
    for t in range(1, num_observations):
        epsilon_t[t] = torch.distributions.Normal(epsilon_t[t - 1].clone(), sigma).rsample()
        dist_epsilon_t = torch.distributions.Normal(epsilon_t[t - 1], sigma)
        ll += dist_epsilon_t.log_prob(epsilon_t[t - 1]).item() #epsilon_t.log_prob(epsilon_t[t - 1])
        #eta_t[t] = epsilon_t[t-1]
    eta_t[1:num_observations] = beta_0 + epsilon_t[0:num_observations-1].clone()
    Rt = torch.exp(eta_t)

    # calculate infections
    for t in range(num_impute, num_observations):
        # Update cumulative newly_infected
        cumulative_infected[t] = cumulative_infected[t - 1].clone() + newly_infected[t - 1].clone()
        # Adjusts for portion of pop that are susceptible
        St[t] = torch.maximum(population.clone() - cumulative_infected[t].clone(), cero) / population.clone()
        # effective number of infectous individuals
        for i in range(0, t - 1):
          effectively_infectious[t] += newly_infected[i].clone() * serial_interval[t - i].clone()
        newly_infected[t] = St[t].clone() * Rt[t].clone() * effectively_infectious[t].clone()

    # calculate expected number of hospitalizations
    expected_daily_hospit[0] = (1e-15) * newly_infected[0].clone()
    for t in range(1, num_observations):
        for i in range(0, t):
            expected_daily_hospit[t] += newly_infected[i].clone() * pi[t - i].clone()
    expected_daily_hospit = alpha * expected_daily_hospit

    # compare observed hospitalizations to model results
    # likelihood of the data wrt. to the model
    dist = 0
    for i in range(1, num_observations):

        p = 1/(1+ expected_daily_hospit[i]/phi)
        dist = torch.distributions.negative_binomial.NegativeBinomial(phi, p-torch.tensor(2.225e-5))
        ll += dist.log_prob(observed_daily_hospit[i])

    ll.backward()

    print(f'k: {k}|| R0:{R0}, grad: {R0.grad}, alpha: {alpha} grad: {alpha.grad}, sigma: {sigma} grad {sigma.grad},'
          f'tau: {tau} grad {tau.grad}, phi: {phi} grad {phi.grad} \n')

    plt.plot(expected_daily_hospit.detach().numpy(), label='expected_daily_hospit')
    plt.plot(observed_daily_hospit.detach().numpy(), label='observed_daily_hospit')
    plt.legend()
    plt.show()

    with torch.no_grad():
        tau -= learning_rate * tau.grad
        phi -= learning_rate * phi.grad
        R0 -= learning_rate * R0.grad
        alpha -= learning_rate * alpha.grad
        sigma -= learning_rate * sigma.grad

        tau.grad = None
        phi.grad = None
        R0.grad = None
        alpha.grad = None
        sigma.grad = None


