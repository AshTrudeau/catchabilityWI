// adding error to independent variable lnPE
data {
  int<lower=0> N;
  vector<lower=0>[N] cpue;
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
  real cpue_q; // coefficient of catchability
  real beta; // coefficient of population density effect on catch
  real epsilon; // outcome noise
  real mu_cpue; // priors for log transformation of cpue
  real<lower=0> sigma_cpue;
}

model {
// transformation
  log(cpue)~normal(mu_cpue, sigma_cpue);
    // Jacobian adjustment for transformation
    // source https://mc-stan.org/docs/2_18/stan-users-guide/changes-of-variables.html
  target += -log(cpue);
  
  // discrete distributions not allowed for latent variables (ugh)
  //true_pop~poisson(pop_lambda);
  true_popDensity~normal(pop_mu, pop_sigma);
  
  popDensity~normal(true_popDensity, pe_sd);
  //popDensity~lognormal(log(to_vector(true_popDensity)^2/sqrt(to_vector(true_popDensity)^2+pe_sd^2)), log(1+(pe_sd^2/(to_vector(true_popDensity))^2)));

  log(cpue)~normal(cpue_q + beta*log(popDensity), epsilon);
  
  cpue_q~normal(0,10);
  beta~normal(0,10);
  //epsilon~uniform(0,10);
  epsilon~cauchy(0,5);
  
  mu_cpue~normal(0,10);
  sigma_cpue~uniform(0,10);
  //sigma_cpue~cauchy(0,5);
  
  //alpha_PE~uniform(0,20);
  //beta_PE~uniform(0,5);
  pop_mu~normal(500, 200);
  pop_sigma~cauchy(0,5);
  


}


