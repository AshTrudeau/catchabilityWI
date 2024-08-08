//
// fitting nonlinear catch equation largemouth bass catch rates



// This model estimates bass population density using mark recapture data from 13 lakes. 
// These estimates are then used as predictors in the catch equation, catch = effort * catchability * population density^beta, 
// linearized to ln(catch) = ln(effort) + ln(catchability) + beta * ln(pop density). ln(catchability) is then broken down
// into three random intercepts by angler, waterbody, and date, with the goal of partitioning variance in catch associated
// with these effects outside of fisheries managers' control, relative to fish population density, which is the traditional
// assumed predictor of catch and under (some) management influence. 

// random effects are unbalanced and not nested

functions {
  int num_zeros(array[] int lmbCatch) {
    int sum=0;
    for(n in 1:size(lmbCatch)){
      sum += (lmbCatch[n] == 0);
    }
    return sum;
  }
}


data {
  // number of observations (205)
  int<lower=1> N;
  // number of anglers (18)
  int<lower=1> A;
  // number of unique dates (42)
  int<lower=1> D;
  // number of lakes (13)
  int<lower=1> L; 
    // number of predictions (all observations)
  // number of predictions for plots
  //int<lower=1> Z; // (234)
  //int<lower=1> Y; // (546)
  //int<lower=1> X; // (169)

  
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
  

 // array[Z] int<lower=1, upper=A> pred_angler;
  //array[Z] int<lower=1, upper=L> pred_angler_pop;
  //array[Y] int<lower=1, upper=D> pred_date;
  //array[Y] int<lower=1, upper=L> pred_date_pop;
  //array[X] int<lower=1, upper=L> pred_lake;
  //array[X] int<lower=1, upper=L> pred_lake_pop;

  
}

transformed data{
  
  int<lower=0> N_zero = num_zeros(lmbCatch);
  array[N - N_zero] int<lower=1> lmbCatch_nonzero;
  int N_nonzero = 0;
  
  for(n in 1:N){
    if (lmbCatch[n]!=0) {
      N_nonzero += 1;
      lmbCatch_nonzero[N_nonzero] = lmbCatch[n];
    }
    
    }
    
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
  real<lower=0, upper=1> theta;
  
  // angler effect
  real<lower=0> sigma_q_a;
  real log_mu_q_a;
  
  // date effect
  real<lower=0> sigma_q_d;
  real log_mu_q_d;
  
  //waterbody/lake effect
  real<lower=0> sigma_q_l;
  real log_mu_q_l;
  
  // for noncentered parameterization
  array[A] real q_a_raw;
  array[D] real q_d_raw;
  array[L] real q_l_raw;
  // population estimate
  vector<lower=0>[L] PE;
  
  //array[N] real zi;


}

transformed parameters{
  // log link prediction
  //array[N] real logCatchHat;
  array[N_zero] real logCatchHat_zero;
  array[N_nonzero] real logCatchHat_nonzero;
  
  array[A] real log_q_a;
  array[D] real log_q_d;
  array[L] real log_q_l;
  
  // for population density
  vector<lower=0>[L] popDensity;
  vector[L] log_popDensity;
  vector[L] log_popDensity_sc;

  
  popDensity = PE ./ surfaceArea;
  log_popDensity = log(popDensity);
  log_popDensity_sc = (log_popDensity-mean(log_popDensity))/sd(log_popDensity);


// note removal of log_mu_q_a/d/l, now wrapped into log_q_mu
// update: divergent transitions problem when I did that; they've been put back in
  for(a in 1:A){
    log_q_a[a] =log_mu_q_a + sigma_q_a * q_a_raw[a];
  }
  for(d in 1:D){
    log_q_d[d] = log_mu_q_d + sigma_q_d * q_d_raw[d];
  }
  for(l in 1:L){
    log_q_l[l] = log_mu_q_l + sigma_q_l * q_l_raw[l];
  }

for(i in 1:N_nonzero){

  logCatchHat_nonzero[i] = log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta * log_popDensity_sc[LL[i]];
}

for(i in 1:N_zero){
    logCatchHat_zero[i] = log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta * log_popDensity_sc[LL[i]];
}

}

model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  target += poisson_lpmf(sumRt | sumCtMt ./ PE);
  //target += normal_lpdf(log_popDensity_sc | 0,1);
  target += lognormal_lpdf(popDensity | 0,2);
  

  target += N_zero * log_sum_exp(log(theta), log1m(theta) + neg_binomial_2_log_lpmf(0 | logCatchHat_zero, phi));
  target += N_nonzero * log1m(theta);
  target += neg_binomial_2_log_lpmf(lmbCatch_nonzero | logCatchHat_nonzero, phi);
  


  target += std_normal_lpdf(q_a_raw);
  target += std_normal_lpdf(q_d_raw);
  target += std_normal_lpdf(q_l_raw);
  
  target += normal_lpdf(log_q_mu | 0,1);


  target += normal_lpdf(log_mu_q_a | 0,1);
  target += normal_lpdf(log_mu_q_d | 0,1);
  target += normal_lpdf(log_mu_q_l | 0,1);
  

  target += exponential_lpdf(sigma_q_a | 1);
  target += exponential_lpdf(sigma_q_d | 1);
  target += exponential_lpdf(sigma_q_l | 1);
  
  target += gamma_lpdf(phi| 1,2);

  target += lognormal_lpdf(beta | -1,1);
  
  target += beta_lpdf(theta | 1,1);
  
}

generated quantities{
  
  // from here: https://discourse.mc-stan.org/t/predicting-from-a-zero-inflated-negative-binomial-model/31434 
  
  //vector[N] log_lik;
  array[N] real posterior_pred_check;
  int<lower=0, upper=1> zero;


  for(n in 1:N){
     //this puts out a probability of "success", which counterintuitively here is zero catch
    zero=bernoulli_rng(theta);
    // swap meaning of zero/not zero with 1-zero * NB prediction
    posterior_pred_check[n]=(1-zero)*neg_binomial_2_log_rng(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity_sc[LL[n]], phi);
  }
  

}

