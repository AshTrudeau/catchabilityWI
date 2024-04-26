// Bayesian population estimate for one lake

data {
  // number of sampling events
  int<lower=0> T;
  // vector of catch at time t * marks out at time t
  array[T] int CtMt;
  // recaps at each sampling event t
  array[T] int Rt;
}

// The parameters accepted by the model. 
parameters {
  // PE can't be an integer because Stan, so real parameter with lower limit of sum of recaps
  real<lower=(sum(Rt))> PE;
}

// Rearranged Schnaebel as Poisson process
model {
  sum(Rt)~poisson(sum(CtMt)/PE);

  PE~gamma(0.001, 0.001);

}

