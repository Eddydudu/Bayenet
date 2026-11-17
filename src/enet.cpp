#include <RcppArmadillo.h>
#include <Rmath.h>
#include "BVCUtilities.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// C++ version of the former R/enet.R: Gibbs sampler for the Bayesian elastic net
// (non-robust, non-sparse). Same model, priors, starting values and update order
// as the R code. "R line" comments refer to the original R/enet.R (Bayenet 0.4).


// One draw from the inverse Gaussian distribution, using the same algorithm as
// VGAM::rinv.gaussian(n = 1, mu, lambda), which the R code called.
static double rinvgauss_vgam(double mu, double lambda)
{
  double u = R::runif(0.0, 1.0);
  double z = R::rnorm(0.0, 1.0);
  double Z = z * z;
  double phi = lambda / mu;
  double y1 = 1.0 - 0.5 * (std::sqrt(Z * Z + 4.0 * phi * Z) - Z) / phi;
  return ((1.0 + y1) * u > 1.0) ? mu / y1 : mu * y1;
}


// [[Rcpp::export()]]
Rcpp::List enet(arma::vec y, arma::mat x, arma::mat c, int maxSteps)
{
  const int n = x.n_rows, p = x.n_cols, q1 = c.n_cols;

  arma::mat XtX = x.t() * x;                       // R line 6
  arma::mat CtC = c.t() * c;                       // used in the Gamma step (R line 48)

  // storage for the samples (R lines 9-14)
  arma::mat betaSamples(maxSteps, p), TauSamples(maxSteps, p), GammaSamples(maxSteps, q1);
  arma::vec sigma2Samples(maxSteps), eta1sample(maxSteps), eta2sample(maxSteps);

  // starting values and hyper-parameters (R lines 22-31)
  arma::vec beta(p, fill::ones);
  arma::vec Tau(p);  Tau.fill(1.5);
  arma::vec Gamma(q1, fill::ones);                 // coefficients of clinical factors
  double sigma2 = 1.0, eta1 = 1.0, eta2 = 1.0;
  const double c1 = 0.1, d1 = 0.1, c2 = 0.1, d2 = 0.1;
  const double tol = 1e-6;                         // same check as MASS::mvrnorm

  for (int k = 0; k < maxSteps; k++) {
    if (k % 100 == 0) Rcpp::checkUserInterrupt();  // lets the user stop a long run

    // ---- sample beta (R lines 38-44) ----
    arma::vec w = Tau / (Tau - 1.0);
    arma::mat invA = arma::inv(XtX + eta2 * arma::diagmat(w));
    arma::vec mean_beta = invA * x.t() * (y - c * Gamma);
    beta = mvrnormCpp(mean_beta, sigma2 * invA, tol);
    betaSamples.row(k) = beta.t();

    // ---- sample Gamma (R lines 46-54); prior covariance is the identity ----
    arma::mat varcov1 = arma::inv(arma::eye(q1, q1) + CtC / sigma2);
    arma::vec res1 = y - x * beta;
    arma::vec mean1 = varcov1 * (c.t() * res1) / sigma2;
    Gamma = mvrnormCpp(mean1, varcov1, tol);
    GammaSamples.row(k) = Gamma.t();

    // ---- sample Tau (R lines 56-68) ----
    double lambda_prime = eta1 / sigma2;
    for (int j = 0; j < p; j++) {
      double nu_prime = std::sqrt(eta1 / (eta2 * beta(j) * beta(j)));
      double temp;
      do {                                         // redraw until positive, as in R
        temp = rinvgauss_vgam(nu_prime, lambda_prime);
        if (std::isnan(temp))
          Rcpp::stop("enet: inverse-Gaussian draw is NaN at step %d (beta[%d] = %g)",
                     k + 1, j + 1, beta(j));
      } while (temp <= 0);
      Tau(j) = 1.0 + 1.0 / temp;
    }
    TauSamples.row(k) = Tau.t();

    // ---- sample sigma2 (R lines 70-83) ----
    arma::vec wNew = Tau / (Tau - 1.0);            // uses the Tau just drawn
    arma::vec r = y - x * beta - c * Gamma;
    double a_temp = n / 2.0 + p;
    double b_temp = 0.5 * arma::accu(arma::square(r))
                  + 0.5 * eta2 * arma::accu(wNew % arma::square(beta))
                  + 0.5 * eta1 * arma::accu(Tau);
    while (true) {
      // MCMCpack::rinvgamma(1, shape = a, scale = b) is 1 / rgamma(1, shape = a, rate = b)
      double z_temp = 1.0 / R::rgamma(a_temp, 1.0 / b_temp);
      double u_temp = R::runif(0.0, 1.0);
      // R: log(u) <= p*log(gamma(1/2)) - p*log(gsl::gamma_inc(1/2, eta1/(2z))).
      // gsl::gamma_inc(1/2, x) = gamma(1/2) * pgamma(x, 1/2, lower = FALSE), so the right side
      // equals -p * log(pgamma(x, 1/2, lower = FALSE)); we compute it on the log scale.
      // NOTE: this right side is always >= 0, so the first draw is always accepted
      // (same behaviour as the R code).
      double rhs = -p * R::pgamma(eta1 / (2.0 * z_temp), 0.5, 1.0, 0, 1);
      if (std::log(u_temp) <= rhs) { sigma2 = z_temp; break; }
    }
    sigma2Samples(k) = sigma2;

    // ---- sample eta1: Metropolis-Hastings step (R lines 85-96) ----
    double temp_shape = p + c1;
    double temp_rate = arma::accu(Tau - 1.0) + 2.0 * sigma2 * d1;
    double temp_eta1 = R::rgamma(temp_shape, 1.0 / temp_rate);
    // log(pgamma(a)/pgamma(b)) computed as a difference of log upper tails (more stable)
    double logr = p * (R::pgamma(eta1, 0.5, 1.0, 0, 1) - R::pgamma(temp_eta1, 0.5, 1.0, 0, 1))
                + p / 2.0 * std::log(eta1 / temp_eta1) + p * (eta1 - temp_eta1);
    logr = std::min(logr, 0.0);
    double u = std::log(R::runif(0.0, 1.0));
    if (u <= logr) eta1 = temp_eta1;
    eta1sample(k) = eta1 * 2.0 * sigma2;

    // ---- sample eta2 (R lines 98-102) ----
    double shape2 = p / 2.0 + c2;
    double rate2 = arma::accu(wNew % arma::square(beta)) / (2.0 * sigma2) + d2;
    eta2 = R::rgamma(shape2, 1.0 / rate2);
    eta2sample(k) = eta2;
  }

  // same names as the list returned by the R code (R line 110)
  return Rcpp::List::create(Rcpp::Named("GS.beta") = betaSamples,
                            Rcpp::Named("sigma2")  = sigma2Samples,
                            Rcpp::Named("tau")     = TauSamples,
                            Rcpp::Named("GS.b")    = GammaSamples,
                            Rcpp::Named("eta1")    = eta1sample,
                            Rcpp::Named("eta2")    = eta2sample);
}
