// adding error to independent variable lnPE
data {
  int<lower=0> N;
  vector<lower=0>[N] fcatch;
  vector<lower=0>[N] effort;
  vector<lower=0>[N] popDensity;
  vector<lower=0>[N] pe_sd; // sd of measurement error of PE
  //vector<lower=0>[N] shoreLength;
  //<lower=0> pe_sd;
}


parameters {
  array[N] real true_popDensity;
  // note: I have multiple lakes with different error terms. I don't want constant measurement
  // error like in the example
  //real alpha_PE;  // priors for 'true' value
  //real beta_PE;
  real<lower=0> pop_mu;
  real<lower=0> pop_sigma;
  real fcatch_q; // coefficient of fcatchability
  real beta; // coefficient of population density effect on fcatch
  real epsilon; // outcome noise
  real mu_fcatch; // priors for log transformation of cpue
  real<lower=0> sigma_fcatch;
}

model {
// transformation
  log(fcatch)~normal(mu_fcatch, sigma_fcatch);
    // Jacobian adjustment for transformation
    // source https://mc-stan.org/docs/2_18/stan-users-guide/changes-of-variables.html
  target += -log(fcatch);
  
  // discrete distributions not allowed for latent variables (ugh)
  //true_pop~poisson(pop_lambda);
  true_popDensity~normal(pop_mu, pop_sigma);
  
  popDensity~normal(true_popDensity, pe_sd);
  //popDensity~lognormal(log(to_vector(true_popDensity)^2/sqrt(to_vector(true_popDensity)^2+pe_sd^2)), log(1+(pe_sd^2/(to_vector(true_popDensity))^2)));

  log(fcatch)~normal(log(effort)+fcatch_q + beta*log(popDensity), epsilon);
  
  fcatch_q~normal(0,10);
  beta~normal(0,10);
  //epsilon~uniform(0,10);
  epsilon~cauchy(0,5);
  
  mu_fcatch~normal(0,10);
  sigma_fcatch~uniform(0,10);
  //sigma_cpue~cauchy(0,5);
  
  //alpha_PE~uniform(0,20);
  //beta_PE~uniform(0,5);
  pop_mu~normal(500, 200);
  pop_sigma~cauchy(0,5);
  


}


