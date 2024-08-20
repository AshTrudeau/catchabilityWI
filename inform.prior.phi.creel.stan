//
// fitting an NB distribution to DNRcreel catch data for largemouth bass


data{
  int<lower=1> N;
  array[N] int<lower=0> lmbCatch;
  vector[N] log_effort;
  
}

parameters{
  real b1;
  real b0;
  real<lower=0> phi;
  
}

model{
  vector[N] logCatchHat;
  
  for(i in 1:N){
    logCatchHat[i] = b1*log_effort[i] +b0;
  }
  
  lmbCatch ~ neg_binomial_2_log(logCatchHat, phi);

  b1~student_t(3,0,1);
  b0~student_t(3,0,1);
  phi~gamma(0,0.5);
  
}

