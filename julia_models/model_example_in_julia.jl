@model function covid19model(
    num_impute,            # [Int] num. of days for which to impute infections
	observed_daily_hospit, # [AbstractVector{<:Int}] reported daily hospitalizations
    π,                     # [AbstractVector{<:Real}] fixed delay distribution from infection to hospitalization
    population,            # [Int] population size
    serial_interval,  	   # [AbstractVector{<:Real}] fixed serial interval from empirical data
	invlink = log,         # inverse link function (see generalized linear model)
	link    = exp,         # link function (see generalized linear model)
)

    num_observations = length(observed_daily_hospit)

    # Latent variables / parameters
    τ         ~ Exponential(1 / 0.03)
    y         ~ truncated(Exponential(τ), 0, 1000) # number of initial newly_infected (seed)
	ϕ         ~ truncated(Normal(0, 5), 0, Inf) # dispersion (shape) parameter for observations
	R0        ~ truncated(Normal(3.6, 0.8), 2, 5) # initial reproduction number
	α         ~ truncated(Normal(1/100,1/100), 0,5/100) # probability to get hospitalized
	σ         ~ truncated(Normal(0.05, 0.03), 0, 0.15) # standart deviation of random walk step

	# initialize time series variables
    newly_infected        = zeros(num_observations) # number of newly infected
	effectively_infectious= zeros(num_observations) # effective number of infectious individuals
    expected_daily_hospit = zeros(num_observations) # expected number of daily hospitalizations
    cumulative_infected   = zeros(num_observations) # cumulative number of infected
    ηt                    = zeros(num_observations) # transformed reproduction number
	ϵt                    = zeros(num_observations) # random walk
	St                    = zeros(num_observations) # fraction of susceptible population

    # seed initial infection / impute first `num_impute` days
    newly_infected[1:num_impute]      .= y
    cumulative_infected[1]             = 0.
    cumulative_infected[2:num_impute] .= cumsum(newly_infected[1:num_impute - 1])
	@. St[1:num_impute]                = max(population - cumulative_infected[1:num_impute], 0) / population

	# calculate Rt: the basic reproduction number
	β0 = invlink(R0)
	ϵt[0] ~ Normal(0, σ)
	for t in 2:num_observations
		ϵt[t] ~ Normal(ϵt[t-1], σ)
		ηt[t] = β0 + ϵt[t] # + RNN[X_t, t]
	end
	Rt = link.(ηt)

	# calculate infections
    for t = (num_impute + 1):num_observations
        # Update cumulative newly_infected
        cumulative_infected[t] = cumulative_infected[t - 1] + newly_infected[t - 1]
        # Adjusts for portion of pop that are susceptible
        St[t] = max(population - cumulative_infected[t], 0) / population
		# effective number of infectous individuals
		effectively_infectious[t] = sum(newly_infected[τ] * serial_interval[t - τ] for τ = 1:t-1)
		# number of new infections (unobserved)
        newly_infected[t] = St[t] * Rt[t] * effectively_infectious[t]
    end

	# calculate expected number of hospitalizations
	expected_daily_hospit[1] = 1e-15 * newly_infected[1]
	for t = 2:num_observations
        expected_daily_hospit[t] = α * sum(newly_infected[τ] * π[t - τ] for τ = 1:t-1)
    end

	# compare observed hospitalizations to model results
	observed_daily_hospit ~ arraydist(NegativeBinomial.(expected_daily_hospit, ϕ))

	return expected_daily_hospit, Rt
end
