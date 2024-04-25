// attempting Schnabel population estimate in Bayesian context

// for now this is for one lake only; can add nesting later

// The input data is a vector 'y' of length 'N'.
data {
  // number of samples
  // vector of catch at time t * marks out at time t
  int<lower=0> T;
  array[T] int CtMt;
  // recaps at each sampling event t
  array[T] int Rt;
}

// The parameters accepted by the model. 
parameters {
  // left off here
  real<lower=(sum(Rt))> PE;
}

// The model to be estimated. 
model {
  sum(Rt)~poisson(sum(CtMt)/PE);

  PE~gamma(0.001, 0.001);
  

}

