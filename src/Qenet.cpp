#include <RcppArmadillo.h>
#include <Rmath.h>
#include "BVCUtilities.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// C++ version of the former R/Qenet.R: Gibbs sampler for the robust (quantile regression)
// Bayesian elastic net, non-sparse. Same model, priors, starting values and update order as
// the R code. "R line" comments refer to the original R/Qenet.R (Bayenet 0.4).
// rinvGauss() in BVCUtilities.cpp uses the same algorithm as SuppDists::rinvGauss
// (Michael, Schucany and Haas, 1976), which the R code called.
// Note: the R function had a default theta = 0.5; robust.R always passes theta explicitly.

// [[Rcpp::export()]]
Rcpp::List Qenet(arma::vec y, arma::mat x, arma::mat c, double theta, int maxSteps)
{
  const int n = x.n_rows, p = x.n_cols, q1 = c.n_cols;

  // storage for the samples (R lines 7-13)
  arma::mat betasample(maxSteps, p), tsample(maxSteps, p), vsample(maxSteps, n),
            Gammasamples(maxSteps, q1);
  arma::vec tausample(maxSteps), eta1sample(maxSteps), eta2sample(maxSteps);

  // constants of the asymmetric Laplace mixture representation (R lines 17-18)
  const double xi1 = (1.0 - 2.0 * theta) / (theta * (1.0 - theta));
  const double xi2 = std::sqrt(2.0 / (theta * (1.0 - theta)));
  const double xi2sq = xi2 * xi2;

  // starting values and hyper-parameters (R lines 19-32)
  arma::vec beta(p, fill::ones);
  arma::vec t(p);  t.fill(2.0);
  arma::vec v(n, fill::ones);
  arma::vec Gamma(q1, fill::ones);
  double tau = 1.0, eta1 = 1.0, eta2 = 1.0;
  const double a = 0.1, b = 0.1, c1 = 0.1, d1 = 0.1, c2 = 0.1, d2 = 0.1, gamma0 = 1.0;
  const int maxTries = 1000;                       // guard for the "redraw until valid" loops

  const arma::mat xsq = arma::square(x), csq = arma::square(c);   // x[,j]^2 and c[,j]^2

  for (int k = 0; k < maxSteps; k++) {
    if (k % 100 == 0) Rcpp::checkUserInterrupt();

    // ---- sample Gamma one coefficient at a time (R lines 36-48) ----
    // rc holds y - c %*% Gamma - x %*% beta - xi1*v; adding c[,j]*Gamma[j] back gives
    // the R vector y_j = y - c[,-j] %*% Gamma[-j] - x %*% beta - xi1*v.
    arma::vec iv = 1.0 / v;
    arma::vec rc = y - c * Gamma - x * beta - xi1 * v;
    for (int j = 0; j < q1; j++) {
      rc += c.col(j) * Gamma(j);
      double invsigma2 = tau * arma::dot(csq.col(j), iv) / xi2sq + 1.0 / gamma0;
      double s2 = 1.0 / invsigma2;
      double mu = tau * arma::accu(rc % c.col(j) % iv) * s2 / xi2sq;
      Gamma(j) = R::rnorm(mu, std::sqrt(s2));
      rc -= c.col(j) * Gamma(j);
    }
    Gammasamples.row(k) = Gamma.t();

    // ---- sample v (R lines 50-62) ----
    arma::vec res = y - x * beta - c * Gamma;
    double lambda = xi1 * xi1 * tau / xi2sq + 2.0 * tau;
    for (int i = 0; i < n; i++) {
      double mu_i = std::sqrt(lambda * xi2sq / (tau * res(i) * res(i)));
      double inv_v;
      int tries = 0;
      do {                                         // R redraws values that are <= 0 or NA
        inv_v = rinvGauss(mu_i, lambda);
        if (++tries > maxTries)
          Rcpp::stop("Qenet: no valid inverse-Gaussian draw for v[%d] at step %d", i + 1, k + 1);
      } while (!(inv_v > 0));
      v(i) = 1.0 / inv_v;
    }
    vsample.row(k) = v.t();

    // ---- sample beta one coefficient at a time (R lines 64-74) ----
    iv = 1.0 / v;                                  // v was just updated
    arma::vec xA = xsq.t() * iv;                   // sum(x[,j]^2 / v) for every j
    arma::vec rb = y - xi1 * v - x * beta - c * Gamma;
    for (int j = 0; j < p; j++) {
      rb += x.col(j) * beta(j);
      double invsigma2 = tau * xA(j) / xi2sq + 2.0 * eta2 * t(j) / (t(j) - 1.0);
      double s2 = 1.0 / invsigma2;
      double mu = tau * arma::accu(rb % x.col(j) % iv) * s2 / xi2sq;
      beta(j) = R::rnorm(mu, std::sqrt(s2));
      rb -= x.col(j) * beta(j);
    }
    betasample.row(k) = beta.t();

    // ---- sample tau (R lines 76-82) ----
    arma::vec rt = y - x * beta - xi1 * v - c * Gamma;
    double rate = arma::accu(arma::square(rt) / (2.0 * xi2sq * v) + v) + b;
    double shape = a + 3.0 * n / 2.0;
    tau = R::rgamma(shape, 1.0 / rate);
    tausample(k) = tau;

    // ---- sample t (R lines 84-95) ----
    double temp_lambda = 2.0 * eta1;
    for (int j = 0; j < p; j++) {
      double temp_nu = std::sqrt(temp_lambda / (2.0 * eta2 * beta(j) * beta(j)));
      double s;
      int tries = 0;
      do {                                         // R redraws values that are <= 0 or NA
        s = rinvGauss(temp_nu, temp_lambda);
        if (++tries > maxTries)
          Rcpp::stop("Qenet: no valid inverse-Gaussian draw for t[%d] at step %d", j + 1, k + 1);
      } while (!(s > 0));
      t(j) = 1.0 / s + 1.0;
    }
    tsample.row(k) = t.t();

    // ---- sample eta1: Metropolis-Hastings step (R lines 97-108) ----
    double temp_shape = p + c1;
    double temp_rate = arma::accu(t - 1.0) + d1;
    double temp_eta1 = R::rgamma(temp_shape, 1.0 / temp_rate);
    // log(pgamma(a)/pgamma(b)) computed as a difference of log upper tails (more stable)
    double logr = p * (R::pgamma(eta1, 0.5, 1.0, 0, 1) - R::pgamma(temp_eta1, 0.5, 1.0, 0, 1))
                + p / 2.0 * std::log(eta1 / temp_eta1) + p * (eta1 - temp_eta1);
    logr = std::min(logr, 0.0);
    if (std::log(R::runif(0.0, 1.0)) <= logr) eta1 = temp_eta1;
    eta1sample(k) = eta1;

    // ---- sample eta2 (R lines 110-114) ----
    double shape2 = p / 2.0 + c2;
    double rate2 = arma::accu(t / (t - 1.0) % arma::square(beta)) + d2;
    eta2 = R::rgamma(shape2, 1.0 / rate2);
    eta2sample(k) = eta2;
  }

  // same names as the list returned by the R code (R line 119)
  return Rcpp::List::create(Rcpp::Named("GS.beta") = betasample,
                            Rcpp::Named("t")       = tsample,
                            Rcpp::Named("tau")     = tausample,
                            Rcpp::Named("GS.b")    = Gammasamples,
                            Rcpp::Named("eta1")    = eta1sample,
                            Rcpp::Named("eta2")    = eta2sample,
                            Rcpp::Named("v")       = vsample);
}
