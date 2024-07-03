//
// fitting nonlinear catch equation largemouth bass catch rates



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
    // number of predictions (all observations)
  // number of predictions for plots
  int<lower=1> Z; // (234)
  int<lower=1> Y; // (546)
  int<lower=1> X; // (169)

  
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
  

  array[Z] int<lower=1, upper=A> pred_angler;
  array[Z] int<lower=1, upper=L> pred_angler_pop;
  array[Y] int<lower=1, upper=D> pred_date;
  array[Y] int<lower=1, upper=L> pred_date_pop;
  array[X] int<lower=1, upper=L> pred_lake;
  array[X] int<lower=1, upper=L> pred_lake_pop;

  
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


}

model {
  
  //for population estimate, Poisson approximation of hypergeometric distribution
  //stan cannot estimate integers as parameters, so can't do hypergeometric dist directly
  target += poisson_lpmf(sumRt | sumCtMt ./ PE);
  target += lognormal_lpdf(popDensity | 0,2);
  
  target += neg_binomial_2_log_lpmf(lmbCatch | logCatchHat, phi);

  target += std_normal_lpdf(q_a_raw);
  target += std_normal_lpdf(q_d_raw);
  target += std_normal_lpdf(q_l_raw);
  
  target += normal_lpdf(log_q_mu | 0,1);
  // testing sensitivity of priors
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
  
 // vector[N] log_lik;
  array[N] real posterior_pred_check;
  array[N] real predictions_all;
  array[L] real predictions_popDensity;
  array[A] real predictions_box_angler;
  array[D] real predictions_box_date;
  array[L] real predictions_box_lake;
  array[Z] real predictions_pop_angler;
  array[Y] real predictions_pop_date;
  array[X] real predictions_pop_lake;
  
  array[N] real predictions_all_link;
  array[N] real predictions_no_popDensity;
  array[N] real predictions_fixed;


  //array[N] real diff;
 // real resid_var;
  real pred_var;
  real pred_var_link;
  real<lower=0> glmm_r2;
  real pred_var_no_popDensity;
  real pred_var_fixed;
  real part_r2_popDensity;
  real prediction_b0;
  real lambda;
  real ICC_adj;
  real ICC_adj_a;
  real ICC_adj_d;
  real ICC_adj_l;
  real fixed_r2;

  for(n in 1:N){
    posterior_pred_check[n]=neg_binomial_2_log_rng(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity[LL[n]],phi);
  }
  
  
  for(n in 1:N){
    predictions_all[n] =exp(log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity[LL[n]]);
  }
  
  for(n in 1:N){
    predictions_all_link[n] = log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]] + beta * log_popDensity[LL[n]];
  }

  // Predicting across population densities (in data) for mean angler, date, and lake effect
  for(l in 1:L){
    predictions_popDensity[l] = exp(log(4) + log_q_mu + log_mu_q_a + log_mu_q_d + log_mu_q_l + beta*log_popDensity[l]);
  }
  // predicting across anglers (all else equal)
  for(a in 1:A){
    predictions_box_angler[a] = exp(log(4) + log_q_mu + log_q_a[a] +log_mu_q_d + log_mu_q_l + beta*log_popDensity[5]);
  }
  // predicting across dates (all else equal)
  for(d in 1:D){
    predictions_box_date[d] = exp(log(4) + log_q_mu + log_mu_q_a +log_q_d[d] + log_mu_q_l + beta*log_popDensity[5]);
  }
  // predicting across lakes (all else equal)
  for(l in 1:L){
    predictions_box_lake[l] = exp(log(4) + log_q_mu + log_mu_q_a +log_mu_q_d + log_q_l[l] + beta*log_popDensity[5]);
  }
  
  // now still predicting across anglers, dates, and lakes, but also across population densities
  // anglers first
  for(z in 1:Z){
    predictions_pop_angler[z] = exp(log(4) + log_q_mu + log_q_a[pred_angler[z]] + log_mu_q_d + log_mu_q_l + beta*log_popDensity[pred_angler_pop[z]]);
  }
  
  for(y in 1:Y){
    predictions_pop_date[y] = exp(log(4) + log_q_mu + log_mu_q_a + log_q_d[pred_date[y]] + log_mu_q_l + beta*log_popDensity[pred_date_pop[y]]);
  }
  
  for(x in 1:X){
    predictions_pop_lake[x] = exp(log(4) + log_q_mu + log_mu_q_a + log_mu_q_d + log_q_l[pred_lake[x]] + beta*log_popDensity[pred_lake_pop[x]]);
  }

  // log_effort is an offset, not really a fixed effect
  // all of these are in the link/latent scale for r2 estimates
  
  for(n in 1:N){
    predictions_no_popDensity[n] = log_effort[n] + log_q_mu + log_q_a[AA[n]] + log_q_d[DD[n]] + log_q_l[LL[n]];
  }
  
  // I think the problems earlier were caused by me forgetting log_mu_q_a, d, l
  // but looking back at Nakagawa et al2017, I really dont think the intercepts are supposedto be there? (though I don't see why it would matter)
  for(n in 1:N){
    predictions_fixed[n] = log_effort[n] + beta*log_popDensity[LL[n]];
  }
  

  prediction_b0 = mean(log_effort) + log_q_mu + log_mu_q_a + log_mu_q_d + log_mu_q_l;


  pred_var = variance(predictions_all);
  pred_var_link = variance(predictions_all_link);
  pred_var_no_popDensity = variance(predictions_no_popDensity);
  pred_var_fixed = variance(predictions_fixed);

  lambda = exp(prediction_b0 + 0.5*(sigma_q_a^2 + sigma_q_d^2 + sigma_q_l^2));

  

  // Nakagawa et al 2017, royal society, most useful paper ever
  glmm_r2=(pred_var_fixed + sigma_q_a^2+sigma_q_d^2+sigma_q_l^2)/(pred_var_fixed + sigma_q_a^2 + sigma_q_d^2 + sigma_q_l^2 + trigamma(((1/lambda)+(1/phi))^-1));
  fixed_r2=(pred_var_fixed)/(pred_var_fixed + sigma_q_a^2 + sigma_q_d^2 + sigma_q_l^2 + trigamma(((1/lambda)+(1/phi))^-1));

// part r2 stands for semi-partial coefficients of determination
// Stoffel et al 2021 got me started, but didn't h ave a method (usable to me) for GLMMs. They cited Jaeger et al 2016, so checking that now
  // both of these are on the link scale
  part_r2_popDensity = (pred_var_link-pred_var_no_popDensity)/(pred_var_link + sigma_q_a^2 + sigma_q_d^2 + sigma_q_l^2  + trigamma(((1/lambda)+(1/phi))^-1));

  // part r2 adapted from Stoffel et al 2021 and Nakagawa et al 2017. May want a statistician to check this

  // partitioning variance explained by random effects
  //ICC
  ICC_adj = (sigma_q_a^2+sigma_q_d^2+sigma_q_l^2)/(sigma_q_a^2+sigma_q_d^2+sigma_q_l^2+trigamma(((1/lambda)+(1/phi))^-1));
  
  ICC_adj_a = (sigma_q_a^2)/(sigma_q_a^2+sigma_q_d^2+sigma_q_l^2+trigamma(((1/lambda)+(1/phi))^-1));
  ICC_adj_d = (sigma_q_d^2)/(sigma_q_a^2+sigma_q_d^2+sigma_q_l^2+trigamma(((1/lambda)+(1/phi))^-1));
  ICC_adj_l = (sigma_q_l^2)/(sigma_q_a^2+sigma_q_d^2+sigma_q_l^2+trigamma(((1/lambda)+(1/phi))^-1));
  

  //VPC simulation method--on data scale, but not confident about how I treated population density
  // instead of using mean population density, now choosing PD from 1 lake

  array[1000] real sim_log_q_a;
  array[1000] real sim_log_q_d;
  array[1000] real sim_log_q_l;

  real mean_log_effort;

  array[1000] real catchHatStar_a;
  array[1000] real catchHatStar_d;
  array[1000] real catchHatStar_l;
  array[1000] real catchHatStar_all;

  array[1000] real var_catchHatStar_all;
  real expect_v1_all;
  
  real var2_catchHatStar_a;
  real var2_catchHatStar_d;
  real var2_catchHatStar_l;
  real var2_catchHatStar_all;

  real<lower=0> vpc_a;
  real<lower=0> vpc_d;
  real<lower=0> vpc_l;
  
  // on the link scale for comparison
    array[1000] real catchHatStar_a_link;
  array[1000] real catchHatStar_d_link;
  array[1000] real catchHatStar_l_link;
  array[1000] real catchHatStar_all_link;

  array[1000] real var_catchHatStar_all_link;
  real expect_v1_all_link;
  
  real var2_catchHatStar_a_link;
  real var2_catchHatStar_d_link;
  real var2_catchHatStar_l_link;
  real var2_catchHatStar_all_link;

  real<lower=0> vpc_a_link;
  real<lower=0> vpc_d_link;
  real<lower=0> vpc_l_link;




 // mean_log_popDensity = mean(log_popDensity);
  mean_log_effort = mean(log_effort);


  for (i in 1:1000){
   sim_log_q_a[i]= normal_rng(log_mu_q_a, sigma_q_a);
   sim_log_q_d[i]= normal_rng(log_mu_q_d, sigma_q_d);
   sim_log_q_l[i]= normal_rng(log_mu_q_l, sigma_q_l);
  }
  
   

  // compute catchHatStar values

 // Switch out prediction lakes here, log_popDensity[X]
  for(i in 1:1000){
    catchHatStar_all[i]= exp(mean_log_effort + log_q_mu + sim_log_q_a[i] + sim_log_q_d[i] + sim_log_q_l[i] + beta*log_popDensity[5]); // simulate all of the random intercepts
    catchHatStar_a[i] = exp(mean_log_effort + log_q_mu + sim_log_q_a[i] + log_mu_q_d + log_mu_q_l + beta*log_popDensity[5]); // simulate one random intercept at a time
    catchHatStar_d[i] = exp(mean_log_effort + log_q_mu + log_mu_q_a +sim_log_q_d[i] + log_mu_q_l + beta*log_popDensity[5]);
    catchHatStar_l[i] = exp(mean_log_effort + log_q_mu + log_mu_q_a + log_mu_q_d + sim_log_q_l[i] + beta*log_popDensity[5]);
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

    expect_v1_all = mean(var_catchHatStar_all);
    
    vpc_a = var2_catchHatStar_a/(var2_catchHatStar_all + expect_v1_all);
    vpc_d = var2_catchHatStar_d/(var2_catchHatStar_all + expect_v1_all);
    vpc_l = var2_catchHatStar_l/(var2_catchHatStar_all + expect_v1_all);

// do it all again but on the link scale for comparison


 // Switch out prediction lakes here, log_popDensity[X]
  for(i in 1:1000){
    catchHatStar_all_link[i]= mean_log_effort + log_q_mu + sim_log_q_a[i] + sim_log_q_d[i] + sim_log_q_l[i] + beta*log_popDensity[5]; // simulate all of the random intercepts
    catchHatStar_a_link[i] = mean_log_effort + log_q_mu + sim_log_q_a[i] + log_mu_q_d + log_mu_q_l + beta*log_popDensity[5]; // simulate one random intercept at a time
    catchHatStar_d_link[i] = mean_log_effort + log_q_mu + log_mu_q_a +sim_log_q_d[i] + log_mu_q_l + beta*log_popDensity[5];
    catchHatStar_l_link[i] = mean_log_effort + log_q_mu + log_mu_q_a + log_mu_q_d + sim_log_q_l[i] + beta*log_popDensity[5];
  }
  
  for(i in 1:1000){
        //level 1 variance
    var_catchHatStar_all_link[i] = catchHatStar_all_link[i] + ((catchHatStar_all_link[i]^2) / phi);

  }
      // level 2 variance
    var2_catchHatStar_a_link = variance(catchHatStar_a_link);
    var2_catchHatStar_d_link = variance(catchHatStar_d_link);
    var2_catchHatStar_l_link = variance(catchHatStar_l_link);
    var2_catchHatStar_all_link = variance(catchHatStar_all_link);

    expect_v1_all_link = mean(var_catchHatStar_all_link);
    
    vpc_a_link = var2_catchHatStar_a_link/(var2_catchHatStar_all_link + expect_v1_all_link);
    vpc_d_link = var2_catchHatStar_d_link/(var2_catchHatStar_all_link + expect_v1_all_link);
    vpc_l_link = var2_catchHatStar_l_link/(var2_catchHatStar_all_link + expect_v1_all_link);

}

