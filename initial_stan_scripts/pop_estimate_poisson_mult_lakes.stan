// attempting Schnabel population estimate in Bayesian context

// Now adding nesting--population estimates for multiple lakes, NO POOLING

//  now adding estimate of population density

data {
  // total number of lakes
  int<lower=1> L;
  // data now has one row per lake instead of one row per sampling event--
  //condensed into sum CtMt and sumRt
  vector[L] sumCtMt;
  // recaps at each sampling event t
  int sumRt[L];
  vector[L] surfaceArea;
}


parameters {
  // I want one estimate of PE for each lake L
  vector<lower=0>[L]  PE;
}

transformed parameters{
  vector<lower=0>[L] popDensity = PE ./ surfaceArea;
}

// The model to be estimated. 
model {
  // continuous distribution of PE because Stan can't estimate an integer param
  // what if it's lognormal?
    //PE~gamma(0.001, 0.001);
    PE~lognormal(0,1);
    sumRt~poisson(sumCtMt ./ PE);
    popDensity~lognormal(0,2);

}

