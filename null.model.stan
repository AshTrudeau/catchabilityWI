//
// fitting nonlinear catch equation to fake data

data {
  // number of observations
  int<lower=1> N;
  // number of anglers
  // all observations of catch (response)
  array[N] int<lower=0> lmbCatch;
  

}


parameters {
  // I want different estimates of q for different anglers, dates, and lakes, as well as an overall mean q
  // without these zero lower bounds, initialization fails


  real<lower=0> phi;


}

model {
  
  target += neg_binomial_2_log_lpmf(lmbCatch | 1, phi);

  target += gamma_lpdf(phi| 1,1);


}

generated quantities{
  vector[N] log_lik;

  
  for(n in 1:N){
    log_lik[n] = neg_binomial_2_log_lpmf(lmbCatch[n] | 1, phi);
  }
  
}

