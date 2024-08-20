//
//

// This is the model that was selected based on LOO CV

// This model estimates bass population density using mark recapture data from 13 lakes. 
// These estimates are then used as predictors in the catch equation, catch = effort * catchability * population density^beta, 
// linearized to ln(catch) = ln(effort) + ln(catchability) + beta * ln(pop density). ln(catchability) is then broken down
// into three random intercepts by angler, waterbody, and date, with the goal of partitioning variance in catch associated
// with these effects outside of fisheries managers' control, relative to fish population density, which is the traditional
// assumed predictor of catch and under (some) management influence. 

// random effects are unbalanced and not nested

// safe neg_binomial generated from here https://discourse.mc-stan.org/t/numerical-stability-of-gps-with-negative-binomial-likelihood/19343/3
// (had problems w ith overflow during posterior predictions)


data {
  // number of observations (205)
  int<lower=1> N;


  // all observations of catch (response)
  array[N] int<lower=0> lmbCatch;
  
  array[N] real log_effort;


}


parameters {
  // I want different estimates of q for different anglers, dates, and lakes, as well as an overall mean q
  // these correspond to differences in angler skill, differences in fishing conditions, and differences in 
  // habitat that may be associated with catch rates (e.g. shoreline structure, depth, habitat complexity)

  // dispersion
  real<lower=0> phi;
  real b1;

}


model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  
  vector[N] logCatchHat;
  for(i in 1:N){
  logCatchHat[i] = b1*log_effort[i];
  }

  lmbCatch ~ neg_binomial_2_log(logCatchHat, phi);
  
  b1 ~ student_t(3,0,1);
  phi ~ gamma(1,5);

}

generated quantities{
  
  vector[N] log_lik;
  array[N] real posterior_pred_check;



   
  for(i in 1:N){
     log_lik[i] = neg_binomial_2_log_lpmf(lmbCatch[i]|log_effort[i], phi);
  }
  for(i in 1:N){
    posterior_pred_check[i] = neg_binomial_2_log_rng(log_effort[i], phi);
  }

}

