import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncexpon, truncnorm, nbinom
import pandas as pd


# TODO: params to fit?

def covid19model(num_impute, observed_daily_hospit, pi, population, serial_interval):

    num_observations = len(observed_daily_hospit)
    # Latent variables / parameters
    tau = np.random.exponential(1 / 0.03)

    # TODO: what does ~ mean (one fixed value or new value each time)
    # b=(upper-lower)/scale, loc=lower, scale=scale
    y = truncexpon.rvs(b=(1000 - 0) / tau, loc=0, scale=tau)  # number of initial newly_infected (seed)

    # For trunc ((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    phi = truncnorm.rvs((0 - 0) / 5, (np.inf - 0) / 5, loc=0, scale=5)  # dispersion (shape) parameter for observations
    R0 = truncnorm.rvs((2 - 3.6) / 0.8, (5 - 3.6) / 0.8, loc=3.6, scale=0.8)  # initial reproduction number
    alpha = truncnorm.rvs((0 - 1/100) / 1/100, (5/100 - 1/100) / 1/100, loc=1/100, scale=1/100)  # probability to get hospitalized
    sigma = truncnorm.rvs((0 - 0.05) / 0.03, (0.15 - 0.05) / 0.03, loc=0.05, scale=0.03)  # standart deviation of random walk step


    # initialize time series variables
    newly_infected = np.zeros(num_observations)  # number of newly infected
    effectively_infectious = np.zeros(num_observations)  # effective number of infectious individuals
    expected_daily_hospit = np.zeros(num_observations)  # expected number of daily hospitalizations
    cumulative_infected = np.zeros(num_observations)  # cumulative number of infected
    eta_t = np.zeros(num_observations)  # transformed reproduction number
    epsilon_t = np.zeros(num_observations)  # random walk
    St = np.zeros(num_observations)  # fraction of susceptible population

    # seed initial infection / impute first `num_impute` days
    newly_infected[0:num_impute] = y
    cumulative_infected[0] = 0.
    cumulative_infected[1:num_impute] = np.cumsum(newly_infected[0:num_impute - 1])
    St[0:num_impute] = np.array([np.maximum(population - x, 0)/population for x in cumulative_infected[0:num_impute]])

    # calculate Rt: the basic reproduction number
    beta_0 = np.log(R0)
    epsilon_t[0] = np.random.normal(0, sigma)
    for t in range(1, num_observations):
        epsilon_t[t] = np.random.normal(epsilon_t[t-1], sigma)
        eta_t[t] = beta_0 + epsilon_t[t]  # + RNN[X_t, t]

    # TODO: ask exponential?
    Rt = np.exp(eta_t)

    # calculate infections
    for t in range(num_impute, num_observations):
        # Update cumulative newly_infected
        cumulative_infected[t] = cumulative_infected[t - 1] + newly_infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        St[t] = np.maximum(population - cumulative_infected[t], 0) / population
        # effective number of infectous individuals
        effectively_infectious[t] = np.sum([newly_infected[i] * serial_interval[t-i] for i in range(0, t-1)])

        # number of new infections (unobserved)
        newly_infected[t] = St[t] * Rt[t] * effectively_infectious[t]

    # calculate expected number of hospitalizations
    # TODO: ask (1e-15)
    expected_daily_hospit[0] = (1e-15) * newly_infected[0]
    for t in range(1, num_observations):
        expected_daily_hospit[t] = alpha * np.sum([newly_infected[i] * pi[t - i] for i in range(0, t - 1)])


    # compare observed hospitalizations to model results
    #observed_daily_hospit ~ arraydist(NegativeBinomial.(expected_daily_hospit, ϕ))
    #expected_daily_hospit_nb = [nbinom.rvs(n, phi, size=1) if n != 0 else 0 for n in expected_daily_hospit]
    # TODO: phi ~ truncated(Normal(0, 5), 0, Inf)-> phi > 1.
    # TODO: What?!? observed_daily_hospit ~ arraydist(NegativeBinomial.(expected_daily_hospit, ϕ))
    plt.plot(expected_daily_hospit, label='expected_daily_hospit')
    #plt.plot(expected_daily_hospit_nb, label='expected_daily_hospit_nb')
    plt.plot(observed_daily_hospit, label='observed_daily_hospit')
    plt.legend()
    plt.show()

    return expected_daily_hospit, Rt


data = pd.read_csv('data/covid19model.csv')

num_impute = 6
observed_daily_hospit = data.hospit.to_numpy()
pi = data.delay_distr.to_numpy()
serial_interval = data.serial_interval.to_numpy()
population = 5793636

for i in range(1):
    expected_daily_hospit, Rt = covid19model(num_impute, observed_daily_hospit, pi, population, serial_interval)