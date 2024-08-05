// This model estimates bass population density using mark recapture data from 13 lakes. 
// These estimates are then used as predictors in the catch equation, catch = effort * catchability * population density^beta, 
// linearized to ln(catch) = ln(effort) + ln(catchability) + beta * ln(pop density). ln(catchability) is then broken down
// into three random intercepts by angler, waterbody, and date, with the goal of partitioning variance in catch associated
// with these effects outside of fisheries managers' control, relative to fish population density, which is the traditional
// assumed predictor of catch and under (some) management influence. 

// update since noncentered.hierarchical.LINEARIZED: random slope by angler on logPopDensity_sc with 
// update since varying.beta.by.angler.Rmd: adding zinf

// random effects are unbalanced and not nested
// working on adding varying effect of angler on beta

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
 
  matrix[2, A] angler_z_raw;         // matrix of intercepts and slopes for each angler
  vector<lower=0>[2] sigma_angler;   // standard deviation for intercept and slope among lakes
  vector[2] angler_global;           // 'global' intercept and slope for angler effect
  cholesky_factor_corr[2] A_corr;    // Cholesky factor of correlation matrix
  
  real mu_date;                     // global mean date
  real mu_lake;                     // global mean lake
  
  real<lower=0> sigma_date;          // sd among dates and lakes around global mean
  real<lower=0> sigma_lake;

  real<lower=0> phi;                 ////likelihood/population deviance
  real<lower=0, upper=1> theta;       // zinf term
  
  // population estimate
  vector<lower=0>[L] PE;
  
  vector[D] date_z_raw;
  vector[L] lake_z_raw;

}

transformed parameters{

  // for population density
  vector<lower=0>[L] popDensity;
  vector[L] log_popDensity;
  vector[L] log_popDensity_sc;

  popDensity = PE ./ surfaceArea;
  log_popDensity = log(popDensity);
  log_popDensity_sc = (log_popDensity-mean(log_popDensity))/sd(log_popDensity);
  
  
  // for model fit
  // this is for angler random intercept and slope (beta)
  matrix[2, A] z_angler;            // angler specific deviation from global mean slope and intercept
  matrix[A, 2] angler_effect;       // angler specific intercepts and slopes
  
  vector[D] date_effect;
  vector[L] lake_effect;
  
  
  z_angler = diag_pre_multiply(sigma_angler, A_corr) * angler_z_raw; 
  
  for(i in 1:A){
    angler_effect[i,1] = angler_global[1] + z_angler[1, i];
    angler_effect[i,2] = angler_global[2] + z_angler[2, i];
  }

  for(d in 1:D){
    date_effect = mu_date + sigma_date * date_z_raw;
  }
  for(l in 1:L){
    lake_effect = mu_lake + sigma_lake * lake_z_raw;
  }
  


}

model {
  
  //array[N] real logCatchHat;
  array[N_zero] real logCatchHat_zero;
  array[N_nonzero] real logCatchHat_nonzero;



  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  
  // population estimate
  sumRt ~ poisson(sumCtMt ./ PE);
  popDensity ~ lognormal(0,1);
  

  // angler_effect matrix: 
    // column 1 is angler-specific intercept
    // column 2 is angler-specific beta
  // lake and date effects are vectors because beta will not be affected by them
  
  A_corr ~ lkj_corr_cholesky(2);
  to_vector(angler_z_raw) ~ normal(0,1);
  date_z_raw ~ normal(0,1);
  lake_z_raw ~ normal(0,1);
  
  angler_global ~ normal(0,1);
  sigma_angler ~ exponential(1);
  
  mu_date ~ normal(0,1);
  mu_lake ~ normal(0,1);
  
  sigma_date ~ exponential(1);
  sigma_lake ~ exponential(1);
  
  phi ~ gamma(1,2);
  
  
  
  for(i in 1:N_nonzero){
    logCatchHat_nonzero[i] = log_effort[i] + angler_effect[AA[i],1] + date_effect[DD[i]] + lake_effect[LL[i]] + angler_effect[AA[i],2] * log_popDensity_sc[LL[i]];
  }
  
    for(i in 1:N_zero){
    logCatchHat_zero[i] = log_effort[i] + angler_effect[AA[i],1] + date_effect[DD[i]] + lake_effect[LL[i]] + angler_effect[AA[i],2] * log_popDensity_sc[LL[i]];
  }

  
  // likelihood
  target += N_zero * log_sum_exp(log(theta), log1m(theta) + neg_binomial_2_log_lpmf(0 | logCatchHat_zero, phi));
  target += N_nonzero * log1m(theta);

  lmbCatch_nonzero ~ neg_binomial_2_log(logCatchHat_nonzero, phi);
  
}

generated quantities{
  
  matrix[2,2] omega;
  array[N] int catch_pred;
  int<lower=0, upper=1> zero;
  
  omega = multiply_lower_tri_self_transpose(A_corr);
  
  for(i in 1:N){
    zero=bernoulli_rng(theta);
    catch_pred[i]=(1-zero)*neg_binomial_2_log_rng(log_effort[i] + angler_effect[AA[i],1] + date_effect[DD[i]] + lake_effect[LL[i]] + angler_effect[AA[i],2] * log_popDensity_sc[LL[i]],phi);
  }


}

