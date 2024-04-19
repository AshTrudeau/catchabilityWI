// adding error to independent variable lnPE
data {
  int<lower=0> N;
  vector<lower=0>[N] cpue;
  vector<lower=0>[N] PE;
  vector<lower=0>[N] pe_sd; // sd of measurement error
  //<lower=0> pe_sd;
}


parameters {
  array[N] real true_PE;
  // note: I have multiple lakes with different error terms. I don't want constant measurement
  // error like in the example
  //real alpha_PE;  // priors for 'true' value
  //real beta_PE;
  real pop_mu;
  real pop_sigma;
  real lnq; // coefficient of catchability
  real beta; // coefficient of population density effect on catch
  real epsilon; // outcome noise
  real mu_cpue; // priors for log transformation of cpue
  real sigma_cpue;
}

model {
// transformation
  log(cpue)~normal(mu_cpue, sigma_cpue);
    // Jacobian adjustment for transformation
    // source https://mc-stan.org/docs/2_18/stan-users-guide/changes-of-variables.html
  target += -log(cpue);
  
  true_PE~normal(pop_mu, pop_sigma);
  //true_PE ~ gamma(alpha_PE, beta_PE);
  // this might not hold up with real data, but it's how I did the simulated data
  PE ~ normal(true_PE, pe_sd);
  

  log(cpue)~normal(lnq + beta*log(PE), epsilon);
  
  lnq~normal(0,1);
  beta~normal(0,1);
  epsilon~uniform(0,10);
  //epsilon~cauchy(0,5);
  
  mu_cpue~normal(0,20);
  sigma_cpue~uniform(0,20);
  //sigma_cpue~cauchy(0,5);
  
  //alpha_PE~uniform(0,20);
  //beta_PE~uniform(0,5);
  pop_mu~normal(0,10);
  pop_sigma~uniform(0,20);
  


}


