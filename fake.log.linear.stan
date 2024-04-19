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

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  vector<lower=0>[N] cpue;
  vector<lower=0>[N] lnPE; 
}


parameters {
  real lnq;
  real beta;
  real epsilon;
  real mu;
  real sigma;
}

model {
// transformation
  log(cpue)~normal(mu, sigma);
    // Jacobian adjustment for transformation
  target += -log(cpue);
  
  log(cpue)~normal(lnq + beta*lnPE, epsilon);
  
  lnq~normal(0,10);
  beta~normal(0,10);
  epsilon~uniform(0,100);
  mu~normal(0,10);
  sigma~normal(0,10);

}


