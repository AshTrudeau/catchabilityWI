//

// This is the model that was selected based on LOO CV

// This model estimates bass population density using mark recapture data from 13 lakes. 
// These estimates are then used as predictors in the catch equation, catch = effort * catchability * population density^beta, 
// linearized to ln(catch) = ln(effort) + ln(catchability) + beta * ln(pop density). ln(catchability) is then broken down
// into three random intercepts by angler, waterbody, and date, with the goal of partitioning variance in catch associated
// with these effects outside of fisheries managers' control, relative to fish population density, which is the traditional
// assumed predictor of catch and under (some) management influence. 

// random effects are unbalanced and not nested

// safe neg_binomial generated from here https://discourse.mc-stan.org/t/numerical-stability-of-gps-with-negative-binomial-likelihood/19343/3
// (had problems w ith overflow during posterior predictions)

functions{
   int neg_binomial_2_log_safe_rng(real eta, real phi) {
    real gamma_rate = gamma_rng(phi, phi / exp(eta));
    if (gamma_rate > exp(20.7)) gamma_rate = exp(20.7); // i think this is the max value before overflow but haven't double checked
    return poisson_rng(gamma_rate);
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
  
  vector[5] z_scores;
  vector[105] z_scores_long;
  vector[84] z_scores_best_worst_angler;
  vector[84] z_scores_best_worst_date;
  vector[21] log_popDensity_pred;
  vector[84] z_scores_medium_angler;
  vector[84] z_scores_medium_date;


  
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
  
  
  // for population density
  vector<lower=0>[L] popDensity;
  vector[L] log_popDensity;
  vector[L] log_popDensity_sc;

  
  popDensity = PE ./ surfaceArea;
  log_popDensity = log(popDensity);
  log_popDensity_sc = (log_popDensity-mean(log_popDensity))/sd(log_popDensity);

// note removal of log_mu_q_a/d/l, now wrapped into log_q_mu
// update: divergent transitions problem when I did that; they've been put back in
// tried again 8/7, this time doing it more correctly. Still had divergent transitions
// forum post out to understand why, contact Chris if that doesn't help
  for(a in 1:A){
    log_q_a[a] =  log_mu_q_a + sigma_q_a * q_a_raw[a];
  }
  for(d in 1:D){
    log_q_d[d] = log_mu_q_d + sigma_q_d * q_d_raw[d];
  }
  for(l in 1:L){
    log_q_l[l] =  log_mu_q_l + sigma_q_l * q_l_raw[l];
  }

for(i in 1:N){

  logCatchHat[i] = log_effort[i] +  log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta * log_popDensity_sc[LL[i]];
}


}

model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  
  sumRt ~ poisson(sumCtMt ./ PE);
  popDensity ~ student_t(3, 0, 50);
  
  lmbCatch ~ neg_binomial_2_log(logCatchHat, phi);
  
  //log_mu_q_a ~ normal(0,1);
  //log_mu_q_d ~ normal(0,1);
  //log_mu_q_l ~ normal(0,1);
    
  log_mu_q_a ~ student_t(3,0,1);
  log_mu_q_d ~ student_t(3,0,1);
  log_mu_q_l ~ student_t(3,0,1);

  
  q_a_raw ~ normal(0,1);
  q_d_raw ~ normal(0,1);
  q_l_raw ~ normal(0,1);
  
  //sigma_q_a ~ exponential(1);
  //sigma_q_d ~ exponential(1);
  //sigma_q_l ~ exponential(1);
  sigma_q_a ~ student_t(3,0,1);
  sigma_q_d ~ student_t(3,0,1);
  sigma_q_l ~ student_t(3,0,1);

  
 // log_q_mu ~ normal(0,1);
 log_q_mu ~ student_t(3,0,1);
  
  phi ~ gamma(1,0.5);
  //beta ~ lognormal(-1, 1);
  beta ~ student_t(3,0,1);

}

generated quantities{
  
  //vector[N] log_lik;
  array[N] int posterior_pred_check;
  array[N] int posterior_pred_random;
  array[N] int posterior_pred_fixed;
  array[N] real link_predictions;
  // model expectation for converting phi of catch NB distribution to sigma
  real prediction_b0;
  real lambda;
  real sigma_2_random;
  real sigma_random;
  
  // for sd of fixed effects only
  vector[N] predict_fixed;
  vector[N] predict_random;
  real sigma_2_fixed;
  real sigma_2_pred;
  real sigma_2_link_pred;
  real sigma_2_total;
  real sigma_2_resid;
  real sigma_fixed;
  real sigma_pred;
  real sigma_link_pred;
  real sigma_total;
  real sigma_resid;
  
  real sigma_post_random;
  real sigma_post_fixed;
  
  real sigma_2_post_random;
  real sigma_2_post_fixed;
  
  real prop_var_explained_pred;
  real prop_angler_vs_fixed_pred;
  real prop_date_vs_fixed_pred;
  real prop_lake_vs_fixed_pred;
  
  real prop_fixed_vs_pred;
  real prop_angler_vs_pred;
  real prop_date_vs_pred;
  real prop_lake_vs_pred;
  

  real r2_marginal;
  real r2_conditional;
  
 //real ICC_a;
 //real ICC_d;
// real ICC_l;
  
  real prior_t_popDensity;
  real prior_t_other;
  real prior_phi;
  
  // for mean catch predictions of observed anglers
  array[A] int predict_angler_catch;
  
  // for mean catch predictions of angler quantiles
  vector[5] quantile_angler;
  array[5] int predict_quantile_catch;
  
  // for mean catch predictions of angler quantiles across population densities
  vector[105] quantile_angler_rep;
  vector[21] log_popDensity_pred_sc;
  vector[105] log_popDensity_sc_rep;
  array[105] int predict_quantile_angler_catch_popDens;
  
  // for mean catch predictions of date quantiles
  
  vector[5] quantile_date;
  array[5] int predict_quantile_date_catch;
  
  // and for mean catch predictions of date quantiles across population densities
  vector[105] quantile_date_rep;
  array[105] int predict_quantile_date_catch_popDens;
  
  // for best worst angler and date
  vector[84] best_worst_angler;
  vector[84] best_worst_date;
  
  vector[84] medium_angler;
  vector[84] medium_date;
  
  vector[84] log_popDensity_sc_best_worst;
  array[84] int predict_best_worst;
  
  array[84] int predict_medium;


  prior_t_popDensity =student_t_rng(3, 0, 50);
  prior_t_other =student_t_rng(3,0,1);
  prior_phi = gamma_rng(1, 0.5);

  // forget it. Instead of r2 values, I'm going to simplify and show graphically the variance in random effects, fixed effect predictions, and predicted total variance (with vertical line for data variance)
  
  // getting 'residual variance' on log scale (observation-specific variance from Nakagawa et al 2017)
  prediction_b0 = mean(log_effort) + log_q_mu;
  
  // does Nakagawa et al 2017 have a typo? it describes sigma tau as the total variance on the latent scale, but then only gives 
  // sigma squared of the random effect? 
  // no  i checked supplement example, it's just the random effects
 lambda = exp(prediction_b0 + 0.5*(sigma_q_a^2+sigma_q_d^2+sigma_q_l^2));
  
  sigma_2_resid = trigamma(((1/lambda)+(1/phi))^(-1));

// I bet the problem is that predict_fixed isn't accounting for the uncertainty in the log_popDensity estimates.
  for(i in 1:N){
  predict_fixed[i] = log_effort[i] + log_q_mu + log_mu_q_a + log_mu_q_d + log_mu_q_l + beta * log_popDensity_sc[LL[i]];
  }
  
    for(i in 1:N){
  predict_random[i] = log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]];
  }

  
    // posterior predictive checks, removed after completion to reduce run time
   for(n in 1:N){
     posterior_pred_check[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity_sc[LL[n]],phi);
   }
   
   for(n in 1:N){
   posterior_pred_random[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]],phi);
   }
   for(n in 1:N){
   posterior_pred_fixed[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_mu_q_a + log_mu_q_d + log_mu_q_l + beta * log_popDensity_sc[LL[n]],phi);
   }
   
   for(n in 1:N){
     link_predictions[n]=log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity_sc[LL[n]];
   }


  sigma_2_fixed = variance(predict_fixed);
  sigma_2_random = variance(predict_random);
  sigma_2_pred = variance(posterior_pred_check);
  sigma_pred = sqrt(sigma_2_pred);
  
  sigma_2_link_pred = variance(link_predictions);
  sigma_link_pred = sqrt(sigma_2_link_pred);
  
  sigma_fixed = sqrt(sigma_2_fixed);
  sigma_random = sqrt(sigma_2_random);
  sigma_pred = sqrt(sigma_2_pred);
  
  sigma_2_random = sigma_q_a^2+sigma_q_d^2+sigma_q_l^2;
  sigma_random=sqrt(sigma_2_random);
  
  sigma_2_total=sigma_2_fixed+sigma_2_random+sigma_2_resid;
  
  sigma_2_post_fixed = variance(posterior_pred_fixed);
  sigma_2_post_random = variance(posterior_pred_random);
  
  sigma_post_fixed =sqrt(sigma_2_post_fixed);
  sigma_post_random = sqrt(sigma_2_post_random);
  
  
  r2_marginal = sigma_2_fixed/sigma_2_total;
  r2_conditional = (sigma_2_fixed+sigma_2_random)/sigma_2_total;
  
  //ICC_a = (sigma_q_a)^2/sigma_2_total;
  //ICC_d = (sigma_q_d)^2/sigma_2_total;
 // ICC_l = (sigma_q_l)^2/sigma_2_total;
  
  prop_var_explained_pred = (sigma_2_random + sigma_2_fixed)/sigma_2_pred;
  prop_angler_vs_fixed_pred = sigma_q_a^2/sigma_2_fixed;
  prop_date_vs_fixed_pred = sigma_q_a^2/sigma_2_fixed;
  prop_lake_vs_fixed_pred = sigma_q_l^2/sigma_2_fixed;
  
  prop_fixed_vs_pred = sigma_2_fixed/sigma_2_pred;
  prop_angler_vs_pred = sigma_q_a^2/sigma_2_pred;
  prop_date_vs_pred = sigma_q_d^2/sigma_2_pred;
  prop_lake_vs_pred = sigma_q_l^2/sigma_2_pred;
  
  sigma_total = sqrt(sigma_2_total);
  sigma_fixed = sqrt(sigma_2_fixed);
  sigma_resid = sqrt(sigma_2_resid);
  
  for(i in 1:A){
  predict_angler_catch[i] = neg_binomial_2_log_safe_rng(mean(log_effort)+log_q_mu+ log_q_a[i] + log_mu_q_d + log_mu_q_l + mean(log_popDensity_sc)*beta, phi);
  }
  
  
  // this gives log_q_a values for 95, 75, 5, 25, and 5 percentiles
  for(i in 1:5){
  quantile_angler[i] = z_scores[i]*sigma_q_a + log_mu_q_a;
  }
  
  for(i in 1:105){
    quantile_angler_rep[i] = z_scores_long[i]*sigma_q_a + log_mu_q_a;
  }
  
  // predict their mean catch of quantiles
  
  for(i in 1:5){
    predict_quantile_catch[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + quantile_angler[i] + log_mu_q_d + log_mu_q_l + mean(log_popDensity_sc)*beta, phi);
  }
  
  // predict mean catch of quantiles across population densities
  
  // first I need to scale the prediction pop densities
  
  for(i in 1:21){
    log_popDensity_pred_sc[i] = (log_popDensity_pred[i]-mean(log_popDensity))/sd(log_popDensity);
  }
  // then repeat it to make it length 55
  
  log_popDensity_sc_rep = append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc,append_row(log_popDensity_pred_sc, log_popDensity_pred_sc))));

  for(i in 1:105){
    predict_quantile_angler_catch_popDens[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + quantile_angler_rep[i] + log_mu_q_d + log_mu_q_l + beta*log_popDensity_sc_rep[i],phi);
  }  
  
  // now catch predictions across daily condition quantiles
    for(i in 1:5){
  quantile_date[i] = z_scores[i]*sigma_q_d + log_mu_q_d;
  }
  
  for(i in 1:105){
    quantile_date_rep[i] = z_scores_long[i]*sigma_q_d + log_mu_q_d;
  }

  for(i in 1:5){
    predict_quantile_date_catch[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + log_mu_q_a + quantile_date[i] + log_mu_q_l + mean(log_popDensity_sc)*beta, phi);
  }

  
    for(i in 1:105){
    predict_quantile_date_catch_popDens[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + log_mu_q_a + quantile_date_rep[i] + log_mu_q_l + beta*log_popDensity_sc_rep[i],phi);
  }  

  // for fun, the best and worst anglers on the best and worst days
  


  for(i in 1:84){
    best_worst_angler[i] = z_scores_best_worst_angler[i]*sigma_q_a + log_mu_q_a;
  }
  
    for(i in 1:84){
    best_worst_date[i] = z_scores_best_worst_date[i]*sigma_q_d + log_mu_q_d;
  }


  log_popDensity_sc_best_worst = append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc,log_popDensity_pred_sc)));

  for(i in 1:84){
    predict_best_worst[i]=neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + best_worst_angler[i] + best_worst_date[i] + log_mu_q_l + beta*log_popDensity_sc_best_worst[i],phi);
  }


  for(i in 1:84){
    medium_angler[i] = z_scores_medium_angler[i]*sigma_q_a + log_mu_q_a;
  }
  
    for(i in 1:84){
    medium_date[i] = z_scores_medium_date[i]*sigma_q_d + log_mu_q_d;
  }
  
  for(i in 1:84){
    predict_medium[i]=neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + medium_angler[i] + medium_date[i] + log_mu_q_l + beta*log_popDensity_sc_best_worst[i],phi);
  }


   
  // for(i in 1:N){
  //   log_lik[i] = neg_binomial_2_log_lpmf(lmbCatch[i]|log_effort[i] + log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + log_q_l[LL[i]] + beta * log_popDensity_sc[LL[i]], phi);
   //}


}

