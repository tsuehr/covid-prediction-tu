using DrWatson
@quickactivate
##
using TransformVariables, LogDensityProblems
using Parameters, Statistics, Distributions
import ReverseDiff # reverse mode automatic differentiation
import ForwardDiff # forward mode automatic differentiation
using CSV
using DataFrames
#----------------------------------------------------------------------------
#some function definitions

"""
NegativeBinomial2(μ, ϕ)

Mean-variance parameterization of `NegativeBinomial`.

## Derivation
`NegativeBinomial` from `Distributions.jl` is parameterized following [1]. With the parameterization in [2], we can solve
for `r` (`n` in [1]) and `p` by matching the mean and the variance given in `μ` and `ϕ`.

We have the following two equations

(1) μ = r (1 - p) / p
(2) μ + μ^2 / ϕ = r (1 - p) / p^2

Substituting (1) into the RHS of (2):
  μ + (μ^2 / ϕ) = μ / p
⟹ 1 + (μ / ϕ) = 1 / p
⟹ p = 1 / (1 + μ / ϕ)
⟹ p = (1 / (1 + μ / ϕ)

Then in (1) we have
  μ = r (1 - (1 / 1 + μ / ϕ)) * (1 + μ / ϕ)
⟹ μ = r ((1 + μ / ϕ) - 1)
⟹ r = ϕ

Hence, the resulting map is `(μ, ϕ) ↦ NegativeBinomial(ϕ, 1 / (1 + μ / ϕ))`.

## References
[1] https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html
[2] https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end
#----------------------------------------------------------------------------
# model definition

# storage for all parameters
@with_kw struct Covid19Model
    num_impute::Int = 6            # [Int] num. of days for which to impute infections
    observed_hospit::Vector{Int64} # [AbstractVector{<:Int}] reported daily hospitalizations
	num_obs::Int               # [Int] number of observations, i.e. length(observed_hospit)
    i2h::Vector{Float64}       # [AbstractVector{<:Real}] fixed delay distribution from infection to hospitalization
    pop::Int = 5793636         # [Int] population size
    si::Vector{Float64}  	   # [AbstractVector{<:Real}] fixed serial interval from empirical data
end

"""
model_transformation(p) returns a function that transformes all parameters in
our model p from a confined space such as
τ ∈ ℝ₊ or y ∈ [0, 1000] to the unconfined space ℝ. This is not necessary but helps
a lot in practice
"""
function model_transformation(model)
	as((
	    τ  = asℝ₊,
	    y  = as(Real,0, 1000),
	    ϕ  = asℝ₊,
	    R0 = as(Real, 2, 5),
	    α  = as(Real, 0, 5/100),
	    σ  = as(Real, 0, 0.15),
		ϵt = as(Array, model.num_obs-1),
	))
end

"""
(model::Covid19Model)(θ) returns the log loglikelihood of our model with
respect to the parameters θ together with inferred values of 'Rt' and
'expected_daily_hospit'.

In a 'standard' machine learning model the loss function is something like
loss(y,ŷ,θ) = sum( (y-ŷ)^2 ) + α*θ,
where 'y' is the observed outcome, 'ŷ' is the model estimate and α is some
regularisation parameter.

In a Bayesian framework the first term is called the log loglikelihood of the
data wrt. the model and the regularisation term is the log likelihood of our
parameters θ wrt. the prior.

The 'standard' loss function is a special case of the Bayesian approach which
assumes Gaussian distribtions.
"""
function evaluate(model::Covid19Model, θ)
	@unpack τ, y, ϕ, R0, α, σ, ϵt = θ
	@unpack num_obs, num_impute, observed_hospit, i2h, pop, si = model

	ℓ = 0. #likelihood

	# log likelihood wrt. our prior ("regularisation")
	ℓ += logpdf( Exponential(1 / 0.03), τ)
	T = typeof(τ)
	ℓ += logpdf( truncated(Exponential(τ), T(0), T(1000)), y) # number of initial newly_infected (seed)
	ℓ += logpdf( truncated(Normal(25, 5), 0, Inf), ϕ) # dispersion (shape) parameter for observations
	ℓ += logpdf( truncated(Normal(3.6, 0.8), 2, 5), R0) # initial reproduction number
	ℓ += logpdf( truncated(Normal(1/100,1/100), 0,5/100), α) # probability to get hospitalized
	ℓ += logpdf( truncated(Normal(0.05, 0.03), 0, 0.15), σ) # standart deviation of random walk step

	newly_infected        = Vector{T}(undef, num_obs) # number of newly infected
	effectively_infectious= Vector{T}(undef, num_obs) # effective number of infectious individuals
	expected_daily_hospit = Vector{T}(undef, num_obs) # expected number of daily hospitalizations
	cumulative_infected   = Vector{T}(undef, num_obs) # cumulative number of infected
	ηt                    = Vector{T}(undef, num_obs) # transformed reproduction number
	St                    = Vector{T}(undef, num_obs) # fraction of susceptible population

	newly_infected[1:num_impute]      .= y
	cumulative_infected[1]             = 0.
	cumulative_infected[2:num_impute] .= cumsum(newly_infected[1:num_impute - 1])
	@. St[1:num_impute]                = max(pop - cumulative_infected[1:num_impute], 0) / pop

	# basic reproduction number as a latent random walk
	β0 = log(R0)
	ηt[1] = β0
	for t in 2:num_obs
		ℓ += logpdf( Normal(ηt[t-1], σ), ϵt[t-1]) # log likelihood wrt. our prior ("regularisation")
		ηt[t] = ϵt[t-1] # + RNN[X_t, t]
	end
	Rt = exp.(ηt)

	# calculate infections
	for t = (num_impute + 1):num_obs
		# Update cumulative newly_infected
		cumulative_infected[t] = cumulative_infected[t - 1] + newly_infected[t - 1]
		# Adjusts for portion of pop that are susceptible
		St[t] = max(pop - cumulative_infected[t], 0) / pop
		# effective number of infectous individuals
		effectively_infectious[t] = sum(newly_infected[τ] * si[t - τ] for τ = 1:t-1)
		# number of new infections (unobserved)
		newly_infected[t] = St[t] * Rt[t] * effectively_infectious[t]
	end

	# calculate expected number of hospitalizations
	expected_daily_hospit[1] = 1e-15 * newly_infected[1]
	for t = 2:num_obs
		expected_daily_hospit[t] = α * sum(newly_infected[τ] * i2h[t - τ] for τ = 1:t-1)
	end

	# compare observed hospitalizations to model results
	# likelihood of the data wrt. to the model
	for i in 1:num_obs
		ℓ += logpdf( NegativeBinomial2(expected_daily_hospit[i], ϕ), observed_hospit[i])
	end

	return (
			loglikelihood = ℓ,
			prediction = (;
				expected_daily_hospit,
				Rt
		)
	)
end

"""
return only the loglikelihood of the model wrt. the parameters θ
"""
(model::Covid19Model)(θ) = evaluate(model, θ).loglikelihood

#----------------------------------------------------------------------------
# initialize the model

data = DataFrame(load(projectdir("covid19model.csv")))

model = Covid19Model(
	num_impute = 6,
	observed_hospit = data.hospit,
	i2h     = data.delay_distr,
	si      = data.serial_interval,
	num_obs = length(data.hospit),
)

# get parameter transformation
t  = model_transformation(model)

# init random parameters from prior distributions
θ = (
	  τ  = rand( Exponential(1 / 0.03) )
	, y  = rand( truncated(Exponential( 30. ), 0, 1000) ) # number of initial newly_infected (seed)
	, ϕ  = rand( truncated(Normal(25, 5), 0, Inf) ) # dispersion (shape) parameter for observations
	, R0 = rand( truncated(Normal(3.6, 0.8), 2, 5) ) # initial reproduction number
	, α  = rand( truncated(Normal(1/100,1/100), 0,5/100) ) # probability to get hospitalized
	, σ  = rand( truncated(Normal(0.05, 0.03), 0, 0.15) ) # standart deviation of random walk step
	, ϵt = rand( MvNormal( zeros(model.num_obs-1), 0.05) )
)
# transformed vector of parameters
x0 = TransformVariables.inverse(t, θ)


transformed_model  = TransformedLogDensity(t, model)
loss(x) = -LogDensityProblems.logdensity(transformed_model, x)
grad(x) = ForwardDiff.gradient(loss, x)
## test:
# you can check that model(θ) ≈ -loss(x0)
model(θ)
loss(x0)
grad(x0)
#---------------------------------------------------------------------------
# update parameters with gradient descent
x = copy(x0)
loss_before_update = loss(x)
x .-= 1e-5 * grad(x)
loss_after_update = loss(x)
# observe that loss_after_update < loss_before_update
#---------------------
# simple optimization scheme

θ_optimal = let
	num_iter = 1000
	ℓs = zeros(num_iter)
	x = copy(x0)
	@progress for i in 1:num_iter
		x .-= 1e-5*grad(x)
		ℓs[i] = loss(x)
	end
	plot(ℓs) |> display
	TransformVariables.transform(t, x)
end

## prediction
p = evaluate(model, θ_optimal).prediction

plot!(p.expected_daily_hospit, label="best fit", title="evaluate fit")
plot!(model.observed_hospit, label="observed")
##
plot(p.Rt, title="basic repoduction number", label="best fit")
