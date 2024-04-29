// attempting Schnabel population estimate in Bayesian context

// Now adding nesting--population estimates for multiple lakes, NO POOLING

data {
  // total number of lakes
  int<lower=1> L;
  // data now has one row per lake instead of one row per sampling event--
  //condensed into sum CtMt and sumRt
  vector[L] sumCtMt;
  // recaps at each sampling event t
 // vector[L] sumRt;
 // array[L] int sumRt;
 int sumRt[L];
}


parameters {
  // I want one estimate of PE for each lake L
  vector<lower=0>[L]  PE;

}

// The model to be estimated. 
model {
    PE~gamma(0.001, 0.001);
    sumRt~poisson(sumCtMt ./ PE);

}

