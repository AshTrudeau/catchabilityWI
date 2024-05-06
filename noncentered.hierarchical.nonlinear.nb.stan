//
// fitting nonlinear catch equation to fake data

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
  array[N] real<lower=0> log_effort;
  // all estimates of population density (covariate)
  array[N] real<lower=0> log_popDensity;
  
}


parameters {
  // I want different estimates of q for different anglers, dates, and lakes, as well as an overall mean q
  // without these zero lower bounds, initialization fails

 
  real log_q_mu;

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

  array[A] real<lower=0> q_a_raw;
  array[D] real<lower=0> q_d_raw;
  array[L] real<lower=0> q_l_raw;

  //array[N] real<lower=0> catchHat;
  array[N] real logCatchHat;
  
  array[A] real log_q_a;
  array[D] real log_q_d;
  array[L] real log_q_l;
  
  
  //array[A] real<lower=0> q_a;
  //array[D] real<lower=0> q_d;
  //array[L] real<lower=0> q_l;
  
 // for(i in 1:N){
  //  catchHat[i] = effort[i] .* (q_mu + q_a[AA[i]] + q_d[DD[i]] + q_l[LL[i]]) .* popDensity[i]^beta;
  //}
  
  for(i in 1:N){
    logCatchHat[i]=log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta*log_popDensity[i];
  }
  
  //logCatchHat = log(catchHat);
  
  //log_q_a = log(q_a);
  //log_q_d = log(q_d);
  //log_q_l = log(q_l);
  
  for(a in 1:A){
    log_q_a[a] = mu_q_a + sigma_q_a * q_a_raw[a];
  }
  for(d in 1:D){
    log_q_d[d] = mu_q_d + sigma_q_d * q_d_raw[d];
  }
  for(l in 1:L){
    log_q_l[l] = mu_q_l + sigma_q_l * q_l_raw[l];
  }

}

model {
  
  lmbCatch~neg_binomial_2_log(logCatchHat, phi);
  

  q_a_raw~std_normal();
  q_d_raw~std_normal();
  q_l_raw~std_normal();
  
  // these need to be above zero
  log_q_mu~normal(0,1);
  
  // these need to be able to include negatives
  mu_q_a~normal(0,1);
  mu_q_d~normal(0,1);
  mu_q_l~normal(0,1);
  
  sigma_q_a~lognormal(0,1);
  sigma_q_d~lognormal(0,1);
  sigma_q_l~lognormal(0,1);
  

  // started out with gamma distribution on phi, tried switching to lognormal to better reflect simulated error
  // resulted in more divergences for some reason? inv_gamma was recommended for log param NB, so landed on that
  phi~inv_gamma(0.4, 0.3);
  beta~lognormal(0,0.5);
  
  
}

