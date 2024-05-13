//
// fitting nonlinear catch equation to fake data

// The input data is a vector 'y' of length 'N'.
data {
  // number of observations
  int<lower=1> N;
  // number of anglers
  int<lower=1> A;
  // number of unique dates
  int<lower=1> D;
  // number of lakes
  int<lower=1> L;
  
  // all observations of catch (response)
  array[N] int<lower=0> lmbCatch;
  
  // indexing observations by anglers, dates, and lakes 
  array[N] int<lower=1, upper=A> AA;
  array[N] int<lower=1, upper=D> DD;
  array[N] int<lower=1, upper=L> LL;

  // all observations of effort (covariate)
  array[N] real<lower=0> effort;
  // all estimates of population density (covariate)
  array[N] real<lower=0> popDensity;
  
}


// phi is inverse overdispersion param
parameters {
  // I want different estimates of q for different anglers, dates, and lakes, as well as an overall mean q
  // without these zero lower bounds, initialization fails
  array[A] real<lower=0> q_a;
  array[D] real<lower=0> q_d;
  array[L] real<lower=0> q_l;
 
  real<lower=0> q_mu;

  real beta;
  real phi;
  
  // hyperparameters
  // lots of divergences if I put zero bounds on sigmas. mu_q values are normally distributed (can include negative values)
  real mu_q_a;
  real sigma_q_a;
  
  real mu_q_d;
  real sigma_q_d;
  
  real mu_q_l;
  real sigma_q_l;
}


transformed parameters{


  for(i in 1:N){
    catchHat[i] = effort[i] .* (q_mu + q_a[AA[i]] + q_d[DD[i]] + q_l[LL[i]]) .* popDensity[i]^beta;

  }
  
  vector[N] catchHat;

}
// use log alternative parameterization for NB? NegBinomial2Log
model {
  
  lmbCatch~neg_binomial_2_log(logCatchHat, phi);
  
  // prior on logCatchHat
  logCatchHat~lognormal(0,1);
  
  // these need to be above zero
  //q_mu~lognormal(0,0.5);
  q_a~lognormal(mu_q_a, sigma_q_a);
  q_d~lognormal(mu_q_d, sigma_q_d);
  q_l~lognormal(mu_q_l, sigma_q_l);
  
  // these need to be able to include negatives
  mu_q_a~normal(0,1);
  mu_q_d~normal(0,1);
  mu_q_l~normal(0,1);
  

  sigma_q_a~exponential(1);
  sigma_q_d~exponential(1);
  sigma_q_l~exponential(1);
  
  // started out with gamma distribution on phi, tried switching to lognormal to better reflect simulated error
  // resulted in more divergences for some reason? 
  //phi~gamma(0.001,0.001);
  phi~inv_gamma(0.4, 0.3);
  beta~lognormal(0,0.5);
  
  
}

