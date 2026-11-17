#include <RcppArmadillo.h>
#include <Rmath.h>
#include "BVCUtilities.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// C++ version of the former R/enetss.R: Gibbs sampler for the Bayesian elastic net with
// spike-and-slab priors (non-robust, sparse). Same model, priors, starting values and update
// order as the R code. "R line" comments refer to the original R/enetss.R (Bayenet 0.4).
// rinvgauss_vgam() and rtgamma_lower() are in BVCUtilities.cpp.

// [[Rcpp::export()]]
Rcpp::List enetss(arma::vec y, arma::mat x, arma::mat c, int maxSteps)
{
  const int n = x.n_rows, p = x.n_cols, q1 = c.n_cols;

  arma::mat CtC = c.t() * c;
  arma::vec xtx = arma::sum(arma::square(x), 0).t();   // t(x[,j]) %*% x[,j] for every j

  // storage for the samples (R lines 9-16)
  arma::mat betaSamples(maxSteps, p), TauSamples(maxSteps, p), SS(maxSteps, p),
            GammaSamples(maxSteps, q1);
  arma::vec sigma2Samples(maxSteps), piSamples(maxSteps), eta1sample(maxSteps),
            eta2sample(maxSteps);

  // starting values and hyper-parameters (R lines 20-36)
  const double lambda1 = 1.0, lambda2 = 1.0;
  arma::vec beta(p, fill::ones);
  arma::vec Tau(p);  Tau.fill(1.5);
  arma::vec Gamma(q1, fill::ones);
  double sigma2 = 1.0;
  const double r = 1.0;
  double u = 1.0;                                  // see the note at the pi step
  double piVal = 0.5;                              // "pi" in the R code
  double eta1 = lambda1 * lambda1 / (4.0 * lambda2);
  double eta2 = lambda2;
  const double c1 = 0.1, d1 = 0.1, c2 = 0.1, d2 = 0.1;
  const double tol = 1e-6;                         // same check as MASS::mvrnorm
  const int maxTries = 1000;                       // guard for the "redraw until valid" loops

  arma::vec z(p), sg(p);

  for (int k = 0; k < maxSteps; k++) {
    if (k % 100 == 0) Rcpp::checkUserInterrupt();
    z.zeros();  sg.zeros();

    // ---- sample beta one coefficient at a time, spike-and-slab (R lines 47-70) ----
    // res holds y - x %*% beta - c %*% Gamma. Adding x[,j]*beta[j] back gives the R residual
    // y - x[,-j] %*% beta[-j] - c %*% Gamma without recomputing it for every j.
    arma::vec res = y - x * beta - c * Gamma;
    for (int j = 0; j < p; j++) {
      res += x.col(j) * beta(j);
      double inv_Tau = Tau(j) * eta2 / (Tau(j) - 1.0);
      double inv_A = 1.0 / (xtx(j) + inv_Tau);
      double xr = arma::dot(x.col(j), res);        // t(x[,j]) %*% res
      double mean = inv_A * xr;
      double var = sigma2 * inv_A;
      double L = inv_A * xr * xr;
      double l0 = piVal + (1.0 - piVal) * std::sqrt(inv_Tau) * std::sqrt(std::abs(inv_A)) *
                  std::exp(L / (2.0 * sigma2));
      double l = piVal / l0;
      u = R::runif(0.0, 1.0);
      if (u <= l) { beta(j) = 0.0;  sg(j) = 0;  z(j) = 0; }
      else        { beta(j) = R::rnorm(mean, std::sqrt(var));  sg(j) = 1;  z(j) = 1; }
      res -= x.col(j) * beta(j);
    }
    betaSamples.row(k) = beta.t();
    SS.row(k) = sg.t();

    // ---- sample Gamma (R lines 72-80); prior covariance is the identity ----
    arma::mat varcov1 = arma::inv(arma::eye(q1, q1) + CtC / sigma2);
    arma::vec mean1 = varcov1 * (c.t() * (y - x * beta)) / sigma2;
    Gamma = mvrnormCpp(mean1, varcov1, tol);
    GammaSamples.row(k) = Gamma.t();

    // ---- sample Tau (R lines 82-109) ----
    double lambda_prime = eta1 / sigma2;
    for (int j = 0; j < p; j++) {
      double temp;
      int tries = 0;
      if (z(j) == 0) {
        // R: hbmem::rtgamma(1, shape = 1/2, scale = 2*sigma2/eta1, a = 1, b = Inf), redrawn
        // until 1 < temp < Inf. rtgamma_lower() is the numerically stable version of it.
        do {
          temp = rtgamma_lower(0.5, 2.0 * sigma2 / eta1, 1.0);
          if (++tries > maxTries)
            Rcpp::stop("enetss: no valid truncated-gamma draw at step %d (sigma2 = %g, eta1 = %g)",
                       k + 1, sigma2, eta1);
        } while (temp <= 1.0 || temp == R_PosInf);
        Tau(j) = temp;
      } else {
        double nu_prime = std::sqrt(eta1 / (eta2 * (beta(j) * beta(j))));   // same order as R
        do {                                       // redraw until positive, as in R
          temp = rinvgauss_vgam(nu_prime, lambda_prime);
          if (std::isnan(temp) || ++tries > maxTries)
            Rcpp::stop("enetss: no valid inverse-Gaussian draw at step %d (beta[%d] = %g)",
                       k + 1, j + 1, beta(j));
        } while (temp <= 0);
        Tau(j) = 1.0 + 1.0 / temp;
      }
    }
    TauSamples.row(k) = Tau.t();

    // ---- sample pi (R lines 111-115) ----
    double sumz = arma::accu(z);
    // NOTE: in the R code the Beta prior parameter u (= 1 at the start, R line 28) is
    // overwritten by "u <- runif(1)" in the beta step (R line 60). So here u is the last
    // uniform draw of the beta step, not 1. We keep this so that C++ and R give the same
    // results. To use the intended prior, replace u by 1.0 below.
    double shape1 = r + p - sumz;
    double shape2_pi = u + sumz;
    piVal = R::rbeta(shape1, shape2_pi);
    piSamples(k) = piVal;

    // ---- sample sigma2 (R lines 118-131) ----
    arma::vec w = Tau / (Tau - 1.0);               // uses the Tau just drawn
    arma::vec resAll = y - x * beta - c * Gamma;
    double a_temp = n / 2.0 + p / 2.0 + sumz / 2.0;
    double b_temp = 0.5 * arma::accu(arma::square(resAll))
                  + 0.5 * eta2 * arma::accu(w % arma::square(beta) % z)
                  + 0.5 * eta1 * arma::accu(Tau);
    while (true) {
      double z_temp = 1.0 / R::rgamma(a_temp, 1.0 / b_temp);
      double u_temp = R::runif(0.0, 1.0);
      // same acceptance rule as the R code (see enet.cpp); the right side is always >= 0
      double rhs = -p * R::pgamma(eta1 / (2.0 * z_temp), 0.5, 1.0, 0, 1);
      if (std::log(u_temp) <= rhs) { sigma2 = z_temp; break; }
    }
    sigma2Samples(k) = sigma2;

    // ---- sample eta1: Metropolis-Hastings step (R lines 133-144) ----
    double temp_shape = p + c1;
    double temp_rate = arma::accu(Tau - 1.0) + 2.0 * sigma2 * d1;
    double temp_eta1 = R::rgamma(temp_shape, 1.0 / temp_rate);
    double logr = p * (R::pgamma(eta1, 0.5, 1.0, 0, 1) - R::pgamma(temp_eta1, 0.5, 1.0, 0, 1))
                + p / 2.0 * std::log(eta1 / temp_eta1) + p * (eta1 - temp_eta1);
    logr = std::min(logr, 0.0);
    if (std::log(R::runif(0.0, 1.0)) <= logr) eta1 = temp_eta1;
    eta1sample(k) = eta1 * 2.0 * sigma2;

    // ---- sample eta2 (R lines 146-150) ----
    double shape2 = c2 + 0.5 * sumz;
    double rate2 = arma::accu(w % arma::square(beta) % z) / (2.0 * sigma2) + d2;
    eta2 = R::rgamma(shape2, 1.0 / rate2);
    eta2sample(k) = eta2;
  }

  // same names as the list returned by the R code (R line 161)
  return Rcpp::List::create(Rcpp::Named("GS.beta") = betaSamples,
                            Rcpp::Named("sigma2")  = sigma2Samples,
                            Rcpp::Named("tau")     = TauSamples,
                            Rcpp::Named("GS.b")    = GammaSamples,
                            Rcpp::Named("pi")      = piSamples,
                            Rcpp::Named("GS.SS")   = SS,
                            Rcpp::Named("eta1")    = eta1sample,
                            Rcpp::Named("eta2")    = eta2sample);
}
