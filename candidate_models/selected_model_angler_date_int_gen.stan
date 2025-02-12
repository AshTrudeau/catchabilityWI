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
  

  // for noncentered parameterization
  array[A] real q_a_raw;
  array[D] real q_d_raw;
  // population estimate
  vector<lower=0>[L] PE;


}

transformed parameters{
  // log link prediction
  array[N] real logCatchHat;
  
  array[A] real log_q_a;
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
// tried again 8/7, this time doing it more correctly. Still had divergent transitions
// forum post out to understand why, contact Chris if that doesn't help
  for(a in 1:A){
    log_q_a[a] =  log_mu_q_a + sigma_q_a * q_a_raw[a];
  }
  for(d in 1:D){
    log_q_d[d] = log_mu_q_d + sigma_q_d * q_d_raw[d];
  }

for(i in 1:N){

  logCatchHat[i] = log_effort[i] +  log_q_mu + log_q_a[AA[i]] + log_q_d[DD[i]] + beta * log_popDensity_sc[LL[i]];
}


}

model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  
  sumRt ~ poisson(sumCtMt ./ PE);
  popDensity ~ student_t(3, 0, 50);
  
  lmbCatch ~ neg_binomial_2_log(logCatchHat, phi);
  

  log_mu_q_a ~ student_t(3,0,1);
  log_mu_q_d ~ student_t(3,0,1);

  
  q_a_raw ~ normal(0,1);
  q_d_raw ~ normal(0,1);

  sigma_q_a ~ student_t(3,0,1);
  sigma_q_d ~ student_t(3,0,1);

  
 // log_q_mu ~ normal(0,1);
 log_q_mu ~ student_t(3,0,1);
  
  phi ~ gamma(1,5);
  beta ~ student_t(3,0,1);

}

generated quantities{
  
  //vector[N] log_lik;
  array[N] int posterior_pred_check;
  array[N] int posterior_pred_angler;
  array[N] int posterior_pred_date;
  array[N] int posterior_pred_fixed;
  array[N] int posterior_pred_nb_only;

  real sigma_post_angler;
  real sigma_post_date;
  real sigma_post_fixed;
  real sigma_post_nb_only;
  real sigma_post_full;
  
  real sigma_2_post_angler;
  real sigma_2_post_date;
  real sigma_2_post_fixed;
  real sigma_2_post_nb_only;
  real sigma_2_post_full;


  real prop_var_angler;
  real prop_var_date;
  real prop_var_popDensity;
  real prop_var_nb;
  

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
  prior_phi = gamma_rng(1, 5);


  
    // posterior predictions for variance partitioning
   for(n in 1:N){
     posterior_pred_check[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] +  beta * log_popDensity_sc[LL[n]],phi);
   }
   
   for(n in 1:N){
   posterior_pred_angler[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_mu_q_d + log_q_a[AA[n]] ,phi);
   }
   
      for(n in 1:N){
   posterior_pred_date[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_mu_q_a +log_q_d[DD[n]] ,phi);
   }

   for(n in 1:N){
   posterior_pred_fixed[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_mu_q_a + log_mu_q_d +  beta * log_popDensity_sc[LL[n]],phi);
   }
   
   for(n in 1:N){
     posterior_pred_nb_only[n]=neg_binomial_2_log_safe_rng(log_effort[n] + log_q_mu + log_mu_q_a + log_mu_q_d, phi);
   }
   



  sigma_2_post_fixed = variance(posterior_pred_fixed);
  sigma_2_post_angler = variance(posterior_pred_angler);
  sigma_2_post_date = variance(posterior_pred_date);
  sigma_2_post_nb_only = variance(posterior_pred_nb_only);
  sigma_2_post_full = variance(posterior_pred_check);
  
  sigma_post_fixed =sqrt(sigma_2_post_fixed);
  sigma_post_angler= sqrt(sigma_2_post_angler);
  sigma_post_date= sqrt(sigma_2_post_date);
  sigma_post_nb_only= sqrt(sigma_2_post_nb_only);
  sigma_post_full = sqrt(sigma_2_post_full);
  

  prop_var_angler = (sigma_2_post_angler-sigma_2_post_nb_only)/sigma_2_post_full;
  prop_var_date = (sigma_2_post_date-sigma_2_post_nb_only)/sigma_2_post_full;
  prop_var_popDensity =(sigma_2_post_fixed-sigma_2_post_nb_only)/sigma_2_post_full;
  prop_var_nb = (sigma_2_post_nb_only)/sigma_2_post_full;

  for(i in 1:A){
  predict_angler_catch[i] = neg_binomial_2_log_safe_rng(mean(log_effort)+log_q_mu+ log_q_a[i] + log_mu_q_d +  beta*mean(log_popDensity_sc), phi);
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
    predict_quantile_catch[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + quantile_angler[i] + log_mu_q_d + beta*mean(log_popDensity_sc), phi);
  }
  
  // predict mean catch of quantiles across population densities
  
  // first I need to scale the prediction pop densities
  
  for(i in 1:21){
    log_popDensity_pred_sc[i] = (log_popDensity_pred[i]-mean(log_popDensity))/sd(log_popDensity);
  }
  // then repeat it to make it length 55
  
  log_popDensity_sc_rep = append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc,append_row(log_popDensity_pred_sc, log_popDensity_pred_sc))));

  for(i in 1:105){
    predict_quantile_angler_catch_popDens[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + quantile_angler_rep[i] + log_mu_q_d +  beta*log_popDensity_sc_rep[i],phi);
  }  
  
  // now catch predictions across daily condition quantiles
    for(i in 1:5){
  quantile_date[i] = z_scores[i]*sigma_q_d + log_mu_q_d;
  }
  
  for(i in 1:105){
    quantile_date_rep[i] = z_scores_long[i]*sigma_q_d + log_mu_q_d;
  }

  for(i in 1:5){
    predict_quantile_date_catch[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + log_mu_q_a + quantile_date[i] + mean(log_popDensity_sc)*beta, phi);
  }

  
    for(i in 1:105){
    predict_quantile_date_catch_popDens[i] = neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + log_mu_q_a + quantile_date_rep[i] +beta*log_popDensity_sc_rep[i],phi);
  }  

  // Compare the best and worst anglers on the best and worst days
  


  for(i in 1:84){
    best_worst_angler[i] = z_scores_best_worst_angler[i]*sigma_q_a + log_mu_q_a;
  }
  
    for(i in 1:84){
    best_worst_date[i] = z_scores_best_worst_date[i]*sigma_q_d + log_mu_q_d;
  }


  log_popDensity_sc_best_worst = append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc, append_row(log_popDensity_pred_sc,log_popDensity_pred_sc)));

  for(i in 1:84){
    predict_best_worst[i]=neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + best_worst_angler[i] + best_worst_date[i] +  beta*log_popDensity_sc_best_worst[i],phi);
  }


  for(i in 1:84){
    medium_angler[i] = z_scores_medium_angler[i]*sigma_q_a + log_mu_q_a;
  }
  
    for(i in 1:84){
    medium_date[i] = z_scores_medium_date[i]*sigma_q_d + log_mu_q_d;
  }
  
  for(i in 1:84){
    predict_medium[i]=neg_binomial_2_log_safe_rng(mean(log_effort) + log_q_mu + medium_angler[i] + medium_date[i] + beta*log_popDensity_sc_best_worst[i],phi);
  }



}

