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

// https://mc-stan.org/docs/stan-users-guide/measurement-error.html 

  // measured population (with error), non-integer for now
  //measurement error (in terms of PE)
data {
  int<lower=0> N;
  vector<lower=0>[N] PE; // measurement of pop
  vector<lower=0>[N] se_pe;  // measurement noise of each PE   
  array[N] int<lower=0, upper=1> catchSomething;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  real beta;
  vector<lower=0>[N] pop;   // unknown true values
  vector<lower=0>[N] pop_mu;             // priors for true value location
  vector<lower=0>[N] pop_sigma;          // scale for true value location

}

// noise is built into the bernoulli model
model {
  // I don't think this is quite right since neither population or PE are really normal
  // Switch to gamma distribution?
  // MOM from here https://math.stackexchange.com/questions/3104688/method-of-moments-with-a-gamma-distribution
  pop~gamma(pop_mu^2/pop_sigma^2, pop_sigma^2/pop_mu); // prior
 // remember that PE is population density, not an integer. 
  PE~gamma(pop^2/se_pe^2, se_pe^2/pop); // measurement model
  
  // propogate 'true' population SD
  vector[N] logPE;
  logPE=log(PE);
  logPE~normal(log(pop), log(se_pe)/pop); //log transforming PE. 
    // Jacobian adjustment for transformation
  target += -logPE;
  
  alpha~normal(0, 3);
  beta~normal(0,3);
  catchSomething ~ bernoulli_logit(alpha+beta*to_vector(log(PE)));
  se_pe~uniform(0,10);
  pop_mu~gamma(10,1);
  pop_sigma~uniform(0, 10);
  
}


