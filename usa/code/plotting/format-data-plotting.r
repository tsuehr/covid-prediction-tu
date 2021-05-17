library(ggplot2)
library(tidyr)
library(dplyr)
library(rstan)
library(data.table)
library(lubridate)
library(gdata)
library(EnvStats)
library(matrixStats)
library(scales)
library(gridExtra)
library(bayesplot)
library(cowplot)


#---------------------------------------------------------------------------
format_data <- function(i, dates, states, estimated_cases_raw, estimated_deaths_raw, 
                        reported_cases, reported_deaths, out, forecast=0, deaths_predicted=estimated_deaths_raw){
  
  N <- length(dates[[i]])
  if(forecast > 0) {
    dates[[i]] = c(dates[[i]], max(dates[[i]]) + 1:forecast)
    N = N + forecast
    reported_cases[[i]] = c(reported_cases[[i]],rep(NA,forecast))
    reported_deaths[[i]] = c(reported_deaths[[i]],rep(NA,forecast))
  }
    
  state <- states[[i]]
  
  estimated_cases <- colMeans(estimated_cases_raw[,1:N,i])
  estimated_cases_li <- colQuantiles(estimated_cases_raw[,1:N,i],prob=.025)
  estimated_cases_ui <- colQuantiles(estimated_cases_raw[,1:N,i],prob=.975)
  estimated_cases_li2 <- colQuantiles(estimated_cases_raw[,1:N,i],prob=.25)
  estimated_cases_ui2 <- colQuantiles(estimated_cases_raw[,1:N,i],prob=.75)
  
  estimated_deaths <- colMeans(estimated_deaths_raw[,1:N,i])
  estimated_deaths_li <- colQuantiles(estimated_deaths_raw[,1:N,i],prob=.025)
  estimated_deaths_ui <- colQuantiles(estimated_deaths_raw[,1:N,i],prob=.975)
  estimated_deaths_li2 <- colQuantiles(estimated_deaths_raw[,1:N,i],prob=.25)
  estimated_deaths_ui2 <- colQuantiles(estimated_deaths_raw[,1:N,i],prob=.75)
  
  deaths_predicted_li <- colQuantiles(deaths_predicted[,1:N,i],prob=.025)
  deaths_predicted_ui <- colQuantiles(deaths_predicted[,1:N,i],prob=.975)
  deaths_predicted_li2 <- colQuantiles(deaths_predicted[,1:N,i],prob=.25)
  deaths_predicted_ui2 <- colQuantiles(deaths_predicted[,1:N,i],prob=.75)
  rt <- colMeans(out$Rt_adj[,1:N,i])
  rt_li <- colQuantiles(out$Rt_adj[,1:N,i],prob=.025)
  rt_ui <- colQuantiles(out$Rt_adj[,1:N,i],prob=.975)
  rt_li2 <- colQuantiles(out$Rt_adj[,1:N,i],prob=.25)
  rt_ui2 <- colQuantiles(out$Rt_adj[,1:N,i],prob=.75)

  data_state_plotting <- data.frame("date" = dates[[i]],
                                    "state" = rep(state, length(dates[[i]])),
                                    "reported_cases" = reported_cases[[i]], 
                                    "predicted_cases" = estimated_cases,
                                    "cases_min" = estimated_cases_li,
                                    "cases_max" = estimated_cases_ui,
                                    "cases_min2" = estimated_cases_li2,
                                    "cases_max2" = estimated_cases_ui2,
                                    "reported_deaths" = reported_deaths[[i]],
                                    "estimated_deaths" = estimated_deaths,
                                    "deaths_min" = estimated_deaths_li,
                                    "deaths_max"= estimated_deaths_ui,
                                    "deaths_min2" = estimated_deaths_li2,
                                    "deaths_max2"= estimated_deaths_ui2,
                                    "deaths_predicted_li"=deaths_predicted_li,
                                    "deaths_predicted_ui"=deaths_predicted_ui,
                                    "deaths_predicted_li2"=deaths_predicted_li2,
                                    "deaths_predicted_ui2"=deaths_predicted_ui2,
                                    "rt" = rt,
                                    "rt_min" = rt_li,
                                    "rt_max" = rt_ui,
                                    "rt_min2" = rt_li2,
                                    "rt_max2" = rt_ui2)
  
  return(data_state_plotting)
  
}
