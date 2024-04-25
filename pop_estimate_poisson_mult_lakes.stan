// attempting Schnabel population estimate in Bayesian context

// Now adding nesting--population estimates for multiple lakes, NO POOLING
// using this example as a guide https://nicholasrjenkins.science/tutorials/bayesian-inference-with-stan/mm_stan/

// The input data is a vector 'y' of length 'N'.
data {
  // total number of lakes
  int<lower=1> L;

  // array (because integer) of catch at sampling event t * marks at large at time t
  array[L] int sumCtMt;
  // recaps at each sampling event t
  array[L] int sumRt;
  
}

parameters {
  // I want one estimate of PE for each lake L
  // where I'm stuck: How do I index lakes with multiple observations? Maybe just calculate it first--one row per lake with sum(CtMt), sum(Rt)
  // int parameters not allowed
  real<lower=sumRt[L]> PE[L];
}

// The model to be estimated. 
model {
  // both of these versions (looped and vectorized) get same 'ill-typed arguments' error
  
// for(i in 1:L){
//    PE[L]~gamma(0.001, 0.001);
//    sumRt[L]~poisson(to_vector(sumCtMt[L])/PE[L]);
  }
    PE~gamma(0.0001, 0.0001);
    sumRt~poisson(to_vector(sumCtMt)/PE)

}

