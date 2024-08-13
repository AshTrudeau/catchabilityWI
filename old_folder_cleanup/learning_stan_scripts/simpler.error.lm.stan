

data {
  int<lower=0> N;
  vector[N] y;
  array[N] real x_meas;   // measurement of x
  vector<lower=0>[N] sd_meas;     // measurement noise
}
parameters {
  array[N] real true_x;    // unknown true value
  real alpha_x;          // prior location
  real beta_x;       // prior scale
  real alpha;
  real beta;
  real sigma;
}
model {
  true_x~gamma(alpha_x, beta_x);
  //true_x ~ normal(mu_x, sigma_x);  // prior
  x_meas ~ normal(true_x, sd_meas);    // measurement model
  y ~ normal(alpha + beta * to_vector(x_meas), sigma);
  
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  
  //mu_x~
  //sigma_x~runif(1,20)
  alpha_x~uniform(0,20);
  beta_x~uniform(0,20);
}
