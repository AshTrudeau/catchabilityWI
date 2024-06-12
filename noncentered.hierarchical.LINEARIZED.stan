//
// fitting nonlinear catch equation largemouth bass catch rates

// 5/15/24 made phi a vector length N--independent error by observation

// This model estimates bass population density using mark recapture data from 13 lakes. 
// These estimates are then used as predictors in the catch equation, catch = effort * catchability * population density^beta, 
// linearized to ln(catch) = ln(effort) + ln(catchability) + beta * ln(pop density). ln(catchability) is then broken down
// into three random intercepts by angler, waterbody, and date, with the goal of partitioning variance in catch associated
// with these effects outside of fisheries managers' control, relative to fish population density, which is the traditional
// assumed predictor of catch and under (some) management influence. 

// random effects are unbalanced and not nested

data {
  // number of observations (205)
  int<lower=1> N;
  // number of anglers (18)
  int<lower=1> A;
  // number of unique dates (42)
  int<lower=1> D;
  // number of lakes (13)
  int<lower=1> L; 
    // number of predictions
  int<lower=1> P;

  
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
  
  array[P] int<lower=1, upper=L> lakeID_pred;
  array[P] int<lower=1, upper=A> anglerID_pred;
  array[P] int<lower=1, upper=D> dateID_pred;
  array[P] real effort_pred;

  
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


}

transformed parameters{
  // log link prediction
  array[N] real logCatchHat;
  
  array[A] real log_q_a;
  array[D] real log_q_d;
  array[L] real log_q_l;
  
  //array[A] real q_a;
  //array[D] real q_d;
  //array[L] real q_l;
  
  // for population density
  vector<lower=0>[L] popDensity;
  vector[L] log_popDensity;
  
  popDensity = PE ./ surfaceArea;
  log_popDensity = log(popDensity);

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

for(i in 1:N){

  logCatchHat[i] = log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta * log_popDensity[LL[i]];
}

 // q_a = exp(log_q_mu+log_q_a);
  //q_d = exp(log_q_mu+log_q_d);
  //q_l = exp(log_q_mu+log_q_l);

}

model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  target += poisson_lpmf(sumRt | sumCtMt ./ PE);
  target += lognormal_lpdf(popDensity | 0,2);
  
  target += neg_binomial_2_log_lpmf(lmbCatch | logCatchHat, phi);

  target += std_normal_lpdf(q_a_raw);
  target += std_normal_lpdf(q_d_raw);
  target += std_normal_lpdf(q_l_raw);
  
  target += normal_lpdf(log_q_mu | 0,1);
  //target += student_t_lpdf(log_q_mu |3, 0,1);


  target += normal_lpdf(log_mu_q_a | 0,1);
  target += normal_lpdf(log_mu_q_d | 0,1);
  target += normal_lpdf(log_mu_q_l | 0,1);
  

  target += exponential_lpdf(sigma_q_a | 1);
  target += exponential_lpdf(sigma_q_d | 1);
  target += exponential_lpdf(sigma_q_l | 1);
  
  target += gamma_lpdf(phi| 1,1);

  target += lognormal_lpdf(beta | -1,1);
  
}

generated quantities{
  
  vector[N] log_lik;
  array[N] real predictions_all;
  array[N] real predictions_fixed_only;
  array[P] real predictions_aslo;


  array[N] real diff;
  real resid_var;
  real pred_var;
  real<lower=0> bayes_r2;
  real pred_fixed_var;
// model has problems if I put a loser limit on fixed_r2 (undefined values for log_lik)
  real fixed_r2;
  real prop_unexp;

  
  for(n in 1:N){
    log_lik[n] = neg_binomial_2_log_lpmf(lmbCatch[n] | log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity[LL[n]], phi);
  }
  
  for(n in 1:N){
    predictions_all[n] =exp(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity[LL[n]]);
  }
  
  for(p in 1:P){
    predictions_aslo[p] = exp(effort_pred[p] + log_q_mu + log_q_a[anglerID_pred[p]] + log_q_d[dateID_pred[p]] + log_q_l[lakeID_pred[p]] + beta * log_popDensity[lakeID_pred[p]]);
  }
  
  for(n in 1:N){
    diff[n] = predictions_all[n] - lmbCatch[n];
  }

  resid_var=variance(diff);
  pred_var = variance(predictions_all);
  
  bayes_r2=pred_var/(pred_var+resid_var);
  
  // variance explained by random/fixed effects only
  // log_effort is an offset, not really a fixed effect
  
  for(n in 1:N){
    predictions_fixed_only[n] = exp(log_effort[n] +log_q_mu + log_mu_q_a +log_mu_q_d + log_mu_q_l +beta*log_popDensity[LL[n]]);
  }
  


  pred_fixed_var = variance(predictions_fixed_only);

  fixed_r2 = (pred_var - pred_fixed_var)/(pred_var+resid_var);
  prop_unexp = resid_var/(pred_var + resid_var);
  
  // part r2 from Stoffel et al 2021. using full and reduced predictions rather than fitting separate full and reduced models

  // partitioning variance explained by random effects
  

  array[1000] real sim_log_q_a;
  array[1000] real sim_log_q_d;
  array[1000] real sim_log_q_l;

  real mean_log_popDensity;
  real mean_log_effort;

  array[1000] real catchHatStar_a;
  array[1000] real catchHatStar_d;
  array[1000] real catchHatStar_l;
  array[1000] real catchHatStar_all;
  //array[1000] real catchHatStar_marg;

  array[1000] real var_catchHatStar_all;
  real expect_v1_all;
  
  real var2_catchHatStar_a;
  real var2_catchHatStar_d;
  real var2_catchHatStar_l;
  real var2_catchHatStar_all;
  //real var2_catchHatStar_marg;

  real<lower=0> vpc_a;
  real<lower=0> vpc_d;
  real<lower=0> vpc_l;
  //real<lower=0> vpc_marg;
  


  mean_log_popDensity = mean(log_popDensity);
  mean_log_effort = mean(log_effort);


  for (i in 1:1000){
   sim_log_q_a[i]= normal_rng(log_mu_q_a, sigma_q_a);
   sim_log_q_d[i]= normal_rng(log_mu_q_d, sigma_q_d);
   sim_log_q_l[i]= normal_rng(log_mu_q_l, sigma_q_l);
  }
  
   

  // compute catchHatStar values

 
  
  for(i in 1:1000){
    catchHatStar_all[i]= exp(mean_log_effort + log_q_mu + sim_log_q_a[i] + sim_log_q_d[i] + sim_log_q_l[i] + beta*mean_log_popDensity); // simulate all of the random intercepts
    catchHatStar_a[i] = exp(mean_log_effort + log_q_mu + sim_log_q_a[i] + log_mu_q_d + log_mu_q_l + beta*mean_log_popDensity); // simulate one random intercept at a time
    catchHatStar_d[i] = exp(mean_log_effort + log_q_mu + log_mu_q_a +sim_log_q_d[i] + log_mu_q_l + beta*mean_log_popDensity);
    catchHatStar_l[i] = exp(mean_log_effort + log_q_mu + log_mu_q_a + log_mu_q_d + sim_log_q_l[i] + beta*mean_log_popDensity);
    // only fixed effects
    // ah, no, this doesn't work (with this simulation method anyway) because the array put out by the following function has var =0 
    //catchHatStar_marg[i] = exp(mean_log_effort + beta*mean_log_popDensity);
  }
  
  for(i in 1:1000){
        //level 1 variance
    var_catchHatStar_all[i] = catchHatStar_all[i] + ((catchHatStar_all[i]^2) / phi);

  }
      // level 2 variance
    var2_catchHatStar_a = variance(catchHatStar_a);
    var2_catchHatStar_d = variance(catchHatStar_d);
    var2_catchHatStar_l = variance(catchHatStar_l);
    var2_catchHatStar_all = variance(catchHatStar_all);
    //var2_catchHatStar_marg = variance(catchHatStar_marg);
    
    expect_v1_all = mean(var_catchHatStar_all);
    
    vpc_a = var2_catchHatStar_a/(var2_catchHatStar_all + expect_v1_all);
    vpc_d = var2_catchHatStar_d/(var2_catchHatStar_all + expect_v1_all);
    vpc_l = var2_catchHatStar_l/(var2_catchHatStar_all + expect_v1_all);
    //vpc_marg = var2_catchHatStar_marg/(var2_catchHatStar_all + expect_v1_all);
  
}

