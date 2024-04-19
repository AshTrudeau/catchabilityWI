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
  vector[N] lnPop;
  array[N] int<lower=0, upper=1> catchSomething;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  real beta;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.

// this is specified with improper (totally uninformed) priors. Add parameter distributions
// below model to have informed priors
model {
  catchSomething ~ bernoulli_logit(alpha+beta*lnPop);
  alpha~normal(0,10);
  beta~normal(0,10);
}


