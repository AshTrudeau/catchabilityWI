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
  array[N] real log_effort;

  // for population estimate
  vector[L] sumCtMt;
  vector[L] surfaceArea;
  int sumRt[L];
  
}


parameters {
  // I want different estimates of q for different anglers, dates, and lakes, as well as an overall mean q
  // without these zero lower bounds, initialization fails

  real log_q_mu;
  real<lower=0> beta;
  real<lower=0> phi;
  
  // hyperparameters
  real mu_q_a;
  real<lower=0> sigma_q_a;
  
  real mu_q_d;
  real<lower=0> sigma_q_d;
  
  real mu_q_l;
  real<lower=0> sigma_q_l;
  
  array[A] real q_a_raw;
  array[D] real q_d_raw;
  array[L] real q_l_raw;
  
  vector<lower=0>[L] PE;


}

transformed parameters{

  array[N] real logCatchHat;
  
  array[A] real log_q_a;
  array[D] real log_q_d;
  array[L] real log_q_l;
  
  array[A] real q_a;
  array[D] real q_d;
  array[L] real q_l;
  
  // for population density
  vector<lower=0>[L] popDensity;
  vector[L] log_popDensity;
  
  popDensity = PE ./ surfaceArea;
  log_popDensity = log(popDensity);


  for(a in 1:A){
    log_q_a[a] = mu_q_a + sigma_q_a * q_a_raw[a];
  }
  for(d in 1:D){
    log_q_d[d] = mu_q_d + sigma_q_d * q_d_raw[d];
  }
  for(l in 1:L){
    log_q_l[l] = mu_q_l + sigma_q_l * q_l_raw[l];
  }

for(i in 1:N){
  logCatchHat[i] = log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta * log_popDensity[LL[i]];
}

  q_a = exp(log_q_a);
  q_d = exp(log_q_d);
  q_l = exp(log_q_l);

}

model {
  
  target += lognormal_lpdf(PE | 0,2);
  target += poisson_lpmf(sumRt | sumCtMt ./ PE);
  target += lognormal_lpdf(popDensity | 0,1);
  
  target += neg_binomial_2_log_lpmf(lmbCatch | logCatchHat, phi);

  target += std_normal_lpdf(q_a_raw);
  target += std_normal_lpdf(q_d_raw);
  target += std_normal_lpdf(q_l_raw);
  
  target += normal_lpdf(log_q_mu | 0,1);


  target += normal_lpdf(mu_q_a | 0,1);
  target += normal_lpdf(mu_q_d | 0,1);
  target += normal_lpdf(mu_q_l | 0,1);
  

  target += exponential_lpdf(sigma_q_a | 5);
  target += exponential_lpdf(sigma_q_d | 5);
  target += exponential_lpdf(sigma_q_l | 5);
  


  //target += exponential_lpdf(phi | 1);
 // target += lognormal_lpdf(phi | 0,1);
  //target += inv_gamma_lpdf(phi | 0.4, 0.3);
  target += gamma_lpdf(phi| 1,1);

  target += lognormal_lpdf(beta | -1,1);
  


}

