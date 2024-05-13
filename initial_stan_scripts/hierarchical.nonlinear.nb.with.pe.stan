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
  // all now replacing direct popDensity input with estimation of popDensity from mark recap
  array[N] real<lower=0> popDensity;
  
}


// phi is inverse overdispersion param
parameters {
  // I want different estimates of q for different anglers, dates, and lakes, as well as an overall mean q
  // I'm taking out the lower bound because it will probably cause problems with the lower q values. It's okay if they overlap zero
  array[A] real<lower=0> q_a;
  array[D] real<lower=0> q_d;
  array[L] real<lower=0> q_l;
 
  real q_mu;

  real beta;
  real<lower=0> phi;
  
  // hyperparameters
  real mu_q_a;
  real sigma_q_a;
  
  real mu_q_d;
  real sigma_q_d;
  
  real mu_q_l;
  real sigma_q_l;
}

transformed parameters{

  vector[N] catchHat;
  vector[N] logCatchHat;
  
  for(i in 1:N){
    catchHat[i] = effort[i] .* (q_mu + q_a[AA[i]] + q_d[DD[i]] + q_l[LL[i]]) .* popDensity[i]^beta;
  }
  
  logCatchHat = log(catchHat);


}
// use log alternative parameterization for NB? NegBinomial2Log
model {
  
  lmbCatch~neg_binomial_2_log(logCatchHat, phi);
  
  q_mu~lognormal(0,0.5);
  q_a~lognormal(mu_q_a, sigma_q_a);
  q_d~lognormal(mu_q_d, sigma_q_d);
  q_l~lognormal(mu_q_l, sigma_q_l);

  mu_q_a~normal(0,1);
  mu_q_d~normal(0,1);
  mu_q_l~normal(0,1);
  
  sigma_q_a~exponential(1);
  sigma_q_d~exponential(1);
  sigma_q_l~exponential(1);
  
  //sigma_q_a~cauchy(0,5);
  //sigma_q_d~cauchy(0,5);
  //sigma_q_l~cauchy(0,5);

  phi~gamma(0.001,0.001);

  beta~lognormal(0,0.5);
  
  
}

