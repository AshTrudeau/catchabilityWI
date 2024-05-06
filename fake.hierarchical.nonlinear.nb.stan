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
  real<lower=0> phi;
  
  // hyperparameters
  real mu_q_a;
  real<lower=0> sigma_q_a;
  
  real mu_q_d;
  real<lower=0> sigma_q_d;
  
  real mu_q_l;
  real<lower=0> sigma_q_l;
}

transformed parameters{

    vector[N] logCatchHat;
  
  for(i in 1:N){
    logCatchHat[i] = log(effort[i] .* (q_mu + q_a[AA[i]] + q_d[DD[i]] + q_l[LL[i]]) .* popDensity[i]^beta);

  }
  

}

model {
  
  //lmbCatch~neg_binomial_2_log(logCatchHat, phi);
 target += neg_binomial_2_log_lpmf(lmbCatch | logCatchHat, phi);

  
  // prior on logCatchHat. Not sure if I need this, but model has divergent transitions either way

  // these need to be above zero
  //q_mu~lognormal(-2,0.5);
  target += lognormal_lpdf(q_mu | -2, 0.5);

  //q_a~lognormal(mu_q_a, sigma_q_a);
  //q_d~lognormal(mu_q_d, sigma_q_d);
  //q_l~lognormal(mu_q_l, sigma_q_l);
  
  target += lognormal_lpdf(q_a | mu_q_a, sigma_q_a);
  target += lognormal_lpdf(q_d | mu_q_d, sigma_q_d);
  target += lognormal_lpdf(q_l | mu_q_l, sigma_q_l);
  
  // these need to be able to include negatives
  //mu_q_a~normal(0,1);
  //mu_q_d~normal(0,1);
  //mu_q_l~normal(0,1);
  
  target += normal_lpdf(mu_q_a| 0, 1);
  target += normal_lpdf(mu_q_d| 0, 1);
  target += normal_lpdf(mu_q_l| 0, 1);
  
  //sigma_q_a~exponential(5);
  //sigma_q_d~exponential(5);
  //sigma_q_l~exponential(5);
  
  target += exponential_lpdf(sigma_q_a | 5);
  target += exponential_lpdf(sigma_q_d | 5);
  target += exponential_lpdf(sigma_q_l | 5);


  // don't switch phi prior to lognormal, it doesn't work
  
  //phi~inv_gamma(0.4, 0.3);
  //beta~lognormal(0,0.5);
  
  target += inv_gamma_lpdf(phi | 0.4, 0.3);
  target += lognormal_lpdf(beta | 0, 0.5);
  
  
}

