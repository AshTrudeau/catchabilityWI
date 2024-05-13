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
  vector<lower=0>[N] popDensity; // measurement of pop
  vector<lower=0>[N] se_pe;  // measurement noise of each PE   
  array[N] int<lower=0, upper=1> catchSomething;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  real beta;
  vector<lower=0>[N] true_pop;   // unknown true values
  vector<lower=0>[N] pop_mu;             // priors for true value location
  vector<lower=0>[N] pop_sigma;          // scale for true value location

}

// noise is built into the bernoulli model
model {
  // I don't think this is quite right since neither population or PE are really normal
  // Switch to gamma distribution?
  // MOM from here https://math.stackexchange.com/questions/3104688/method-of-moments-with-a-gamma-distribution
  true_pop~normal(pop_mu, pop_sigma); // prior
 // remember that PE is population density, not an integer. 
  popDensity~normal(true_pop, se_pe); // measurement model
  
  // propogate 'true' population SD
  //vector[N] logPE;
  //logPopDensity=log(popDensity);
  //logPopDensity~normal(log(pop), log(se_pe)/pop); //log transforming PE. 
    // Jacobian adjustment for transformation
  //target += -logPE;
  
    catchSomething ~ bernoulli_logit(alpha+beta*to_vector(log(popDensity)));

  
  alpha~normal(0, 10);
  beta~normal(0,10);
  se_pe~cauchy(0,5);
  pop_mu~normal(500,200);
  pop_sigma~cauchy(0, 5);
  
}


