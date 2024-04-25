// attempting Schnabel population estimate in Bayesian context

// for now this is for one lake only; can add nesting later

// The input data is a vector 'y' of length 'N'.
data {
  // number of samples
  int<lower=1> C;
  // vector of marks at large prior to each sample t
  int<lower=1> M;
  // recaps at each sampling event t

  int<lower=0> sumRt;
}

// The parameters accepted by the model. 
parameters {
  // left off here
  real<lower=(M)> PE;
}

// The model to be estimated. 
model {
  sumRt~poisson((C*M)/PE);

  PE~gamma(0.001, 0.001);
  

}

