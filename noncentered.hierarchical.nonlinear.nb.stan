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

  array[A] real<lower=0> q_a_raw;
  array[D] real<lower=0> q_d_raw;
  array[L] real<lower=0> q_l_raw;
 
  real<lower=0> q_mu;

  real beta;
  real phi;
  
  // hyperparameters
  real mu_q_a;
  real<lower=0> sigma_q_a;
  
  real mu_q_d;
  real<lower=0> sigma_q_d;
  
  real mu_q_l;
  real<lower=0> sigma_q_l;
}

transformed parameters{

  array[N] real<lower=0> catchHat;
  array[N] real logCatchHat;
  
  array[A] real log_q_a;
  array[D] real log_q_d;
  array[L] real log_q_l;
  
  array[A] real<lower=0> q_a;
  array[D] real<lower=0> q_d;
  array[L] real<lower=0> q_l;
  
  for(i in 1:N){
    catchHat[i] = effort[i] .* (q_mu + q_a[AA[i]] + q_d[DD[i]] + q_l[LL[i]]) .* popDensity[i]^beta;
  }
  
  logCatchHat = log(catchHat);
  
  log_q_a = log(q_a);
  log_q_d = log(q_d);
  log_q_l = log(q_l);
  
  // this is directly based off a working example; why was there no a index in the loop?
  // I added them in what I figured was the appropriate place
  for(a in 1:A){
    log_q_a[A] = mu_q_a + sigma_q_a * q_a_raw[A];
  }
  for(d in 1:D){
    log_q_d[D] = mu_q_d + sigma_q_d * q_d_raw[D];
  }
  for(l in 1:L){
    log_q_l[L] = mu_q_l + sigma_q_l * q_l_raw[L];
  }

}

model {
  
  lmbCatch~neg_binomial_2_log(logCatchHat, phi);
  
  // prior on logCatchHat. Not sure if I need this, but model has divergent transitions either way
  logCatchHat~lognormal(0,1);
  
  q_a_raw~std_normal();
  q_d_raw~std_normal();
  q_l_raw~std_normal();
  
  // these need to be above zero
  q_mu~lognormal(0,0.5);
  
  // these need to be able to include negatives
  mu_q_a~normal(0,1);
  mu_q_d~normal(0,1);
  mu_q_l~normal(0,1);
  
  sigma_q_a~exponential(1);
  sigma_q_d~exponential(1);
  sigma_q_l~exponential(1);
  
  // started out with gamma distribution on phi, tried switching to lognormal to better reflect simulated error
  // resulted in more divergences for some reason? inv_gamma was recommended for log param NB, so landed on that
  //phi~gamma(0.001,0.001);
  //phi~lognormal(0,0.5);
  phi~inv_gamma(0.4, 0.3);
  beta~lognormal(0,0.5);
  
  
}

