//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
  int<lower=0> N;
  array[N] real<lower=0> PE; 
  vector<lower=0>[N] se_pe;    
  array[N] int<lower=0, upper=1> catchSomething;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  real beta;
  real<lower=0> pop;   // unknown true value
  real pe_mu;    
  real pe_sigma;
  real mu;
  real sigma;
}

// noise is built into the bermoulli model
model {
  // I don't think this is quite right since neither population or PE are really normal
  // Switch to gamma distribution?
  // MOM from here https://math.stackexchange.com/questions/3104688/method-of-moments-with-a-gamma-distribution
  pop~gamma(pe_mu^2/pe_sigma^2, pe_sigma^2/pe_mu); // prior
  PE~gamma(pop^2/se_pe^2, se_pe^2/pop); // measurement model
  
  
  log(PE)~normal(mu, sigma); //log transforming PE
    // Jacobian adjustment for transformation
  target += -log(PE);
  
  alpha~normal(0, 10);
  beta~normal(0,10);
  catchSomething ~ bernoulli_logit(alpha+beta*to_vector(log(PE)));
  
}

