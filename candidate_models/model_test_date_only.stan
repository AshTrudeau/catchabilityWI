//



// For model comparison, this is model with only angler effect

data {
  // number of observations (205)
  int<lower=1> N;
  // number of anglers (18)
  // number of unique dates (42)
  int<lower=1> D;
  // number of lakes (13)
  int<lower=1> L; 

  
  // all observations of catch (response)
  array[N] int<lower=0> lmbCatch;
  
  // indexing observations by anglers, dates, and lakes 
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
  // these correspond to differences in angler skill, differences in fishing conditions, and differences in 
  // habitat that may be associated with catch rates (e.g. shoreline structure, depth, habitat complexity)

  real log_q_mu;
  // beta is the hyperstability coefficient--degree of nonlinearity in response of catch to population density
  real<lower=0> beta;
  // dispersion
  real<lower=0> phi;
  

  // date effect
  real<lower=0> sigma_q_d;
  real log_mu_q_d;

  // for noncentered parameterization
  array[D] real q_d_raw;
  // population estimate
  vector<lower=0>[L] PE;


}

transformed parameters{
  // log link prediction
  array[N] real logCatchHat;
  
  array[D] real log_q_d;

  // for population density
  vector<lower=0>[L] popDensity;
  vector[L] log_popDensity;
  vector[L] log_popDensity_sc;

  
  popDensity = PE ./ surfaceArea;
  log_popDensity = log(popDensity);
  log_popDensity_sc = (log_popDensity-mean(log_popDensity))/sd(log_popDensity);

// note removal of log_mu_q_a/d/l, now wrapped into log_q_mu
// update: divergent transitions problem when I did that; they've been put back in
  for(d in 1:D){
    log_q_d[d] = log_mu_q_d + sigma_q_d * q_d_raw[d];
  }

for(i in 1:N){

  logCatchHat[i] = log_effort[i] + log_q_mu +log_q_d[DD[i]]  + beta * log_popDensity_sc[LL[i]];
}


}

model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  target += poisson_lpmf(sumRt | sumCtMt ./ PE);
  target += lognormal_lpdf(popDensity | 0,2);
  
  target += neg_binomial_2_log_lpmf(lmbCatch | logCatchHat, phi);

  target += std_normal_lpdf(q_d_raw);

  target += normal_lpdf(log_q_mu | 0,1);
  // testing sensitivity of priors
  //target += student_t_lpdf(log_q_mu |3, 0,1);


  target += normal_lpdf(log_mu_q_d | 0,1);


  target += exponential_lpdf(sigma_q_d | 1);

  target += gamma_lpdf(phi| 1,2);

  target += lognormal_lpdf(beta | -1,1);
  
}

generated quantities{
  
  vector[N] log_lik;
  array[N] real posterior_pred_check;


  for(n in 1:N){
    posterior_pred_check[n]=neg_binomial_2_log_rng(log_effort[n] + log_q_mu +  log_q_d[DD[n]] +  beta * log_popDensity_sc[LL[n]],phi);
  }
  
  for(i in 1:N){
    log_lik[i] = neg_binomial_2_log_lpmf(lmbCatch[i]|log_effort[i] + log_q_mu +  log_q_d[DD[i]] +  beta * log_popDensity_sc[LL[i]], phi);
  }
  
}
