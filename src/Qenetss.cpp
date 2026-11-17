#include <RcppArmadillo.h>
#include <Rmath.h>
#include "BVCUtilities.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// C++ version of the former R/Qenetss.R: Gibbs sampler for the robust (quantile regression)
// Bayesian elastic net with spike-and-slab priors. Same model, priors, starting values and
// update order as the R code. "R line" comments refer to the original R/Qenetss.R (Bayenet 0.4).
// rinvGauss() (same algorithm as SuppDists::rinvGauss) and rtgamma_lower() (numerically
// stable version of hbmem::rtgamma) are in BVCUtilities.cpp.
// Note: the R function had a default theta = 0.5; robust.R always passes theta explicitly.

// [[Rcpp::export()]]
Rcpp::List Qenetss(arma::vec y, arma::mat x, arma::mat c, double theta, int maxSteps)
{
  const int n = x.n_rows, p = x.n_cols, q1 = c.n_cols;

  // storage for the samples (R lines 7-15)
  arma::mat betasample(maxSteps, p), tsample(maxSteps, p), vsample(maxSteps, n),
            SS(maxSteps, p), Gammasamples(maxSteps, q1);
  arma::vec tausample(maxSteps), eta1sample(maxSteps), eta2sample(maxSteps),
            pisample(maxSteps);

  // constants of the asymmetric Laplace mixture representation (R lines 18-19)
  const double xi1 = (1.0 - 2.0 * theta) / (theta * (1.0 - theta));
  const double xi2 = std::sqrt(2.0 / (theta * (1.0 - theta)));
  const double xi2sq = xi2 * xi2;

  // starting values and hyper-parameters (R lines 20-36)
  arma::vec beta(p, fill::ones);
  arma::vec t(p);  t.fill(2.0);
  arma::vec v(n, fill::ones);
  arma::vec Gamma(q1, fill::ones);
  double tau = 1.0, eta1 = 1.0, eta2 = 1.0;
  const double a = 0.1, b = 0.1, c1 = 0.1, d1 = 0.1, c2 = 0.1, d2 = 0.1, gamma0 = 1.0;
  const double r1 = 1.0, u1 = 1.0;
  double piVal = 0.5;                              // "pi" in the R code
  const int maxTries = 1000;                       // guard for the "redraw until valid" loops

  const arma::mat xsq = arma::square(x), csq = arma::square(c);   // x[,j]^2 and c[,j]^2
  arma::vec z(p), sg(p);

  for (int k = 0; k < maxSteps; k++) {
    if (k % 100 == 0) Rcpp::checkUserInterrupt();

    // ---- sample Gamma one coefficient at a time (R lines 40-52) ----
    // NOTE: the R code divides by xi2 here (R lines 44 and 48), while Qenet.R and the beta
    // step below divide by xi2^2. We keep xi2 so that C++ and R give the same results.
    arma::vec iv = 1.0 / v;
    arma::vec rc = y - c * Gamma - x * beta - xi1 * v;
    for (int j = 0; j < q1; j++) {
      rc += c.col(j) * Gamma(j);
      double invsigma2 = tau * arma::dot(csq.col(j), iv) / xi2 + 1.0 / gamma0;
      double s2 = 1.0 / invsigma2;
      double mu = tau * arma::accu(rc % c.col(j) % iv) * s2 / xi2;
      Gamma(j) = R::rnorm(mu, std::sqrt(s2));
      rc -= c.col(j) * Gamma(j);
    }
    Gammasamples.row(k) = Gamma.t();

    // ---- sample v (R lines 54-66) ----
    arma::vec res = y - x * beta - c * Gamma;
    double lambda = xi1 * xi1 * tau / xi2sq + 2.0 * tau;
    for (int i = 0; i < n; i++) {
      double mu_i = std::sqrt(lambda * xi2sq / (tau * res(i) * res(i)));
      double inv_v;
      int tries = 0;
      do {                                         // R redraws values that are <= 0 or NA
        inv_v = rinvGauss(mu_i, lambda);
        if (++tries > maxTries)
          Rcpp::stop("Qenetss: no valid inverse-Gaussian draw for v[%d] at step %d", i + 1, k + 1);
      } while (!(inv_v > 0));
      v(i) = 1.0 / inv_v;
    }
    vsample.row(k) = v.t();

    // ---- sample beta one coefficient at a time, spike-and-slab (R lines 68-90) ----
    z.zeros();  sg.zeros();
    iv = 1.0 / v;                                  // v was just updated
    arma::vec xA = xsq.t() * iv;                   // sum(x[,j]^2 / v) for every j
    arma::vec rb = y - xi1 * v - x * beta - c * Gamma;
    for (int j = 0; j < p; j++) {
      rb += x.col(j) * beta(j);
      double prior_prec = 2.0 * eta2 * t(j) / (t(j) - 1.0);
      double invsigma2 = tau * xA(j) / xi2sq + prior_prec;
      double s2 = 1.0 / invsigma2;
      double Bsum = arma::accu(rb % x.col(j) % iv);          // sum(B)
      double mu = tau * Bsum * s2 / xi2sq;
      double BB = tau * Bsum / xi2sq;
      double d = std::sqrt(prior_prec) * std::sqrt(s2) * std::exp(0.5 * s2 * BB * BB);
      double l = piVal / (piVal + (1.0 - piVal) * d);
      double u = R::runif(0.0, 1.0);
      if (u < l) { beta(j) = 0.0;  z(j) = 0;  sg(j) = 0; }
      else       { beta(j) = R::rnorm(mu, std::sqrt(s2));  z(j) = 1;  sg(j) = 1; }
      rb -= x.col(j) * beta(j);
    }
    betasample.row(k) = beta.t();
    SS.row(k) = sg.t();

    // ---- sample tau (R lines 92-98) ----
    arma::vec rt = y - x * beta - xi1 * v - c * Gamma;
    double rate = arma::accu(arma::square(rt) / (2.0 * xi2sq * v) + v) + b;
    double shape = a + 3.0 * n / 2.0;
    tau = R::rgamma(shape, 1.0 / rate);
    tausample(k) = tau;

    // ---- sample t (R lines 100-121) ----
    for (int j = 0; j < p; j++) {
      int tries = 0;
      if (beta(j) == 0) {
        // R: hbmem::rtgamma(1, shape = 1/2, scale = 1/eta1, a = 1, b = Inf), redrawn until
        // 1 < temp.t < Inf. rtgamma_lower() is the numerically stable version of it.
        double temp_t;
        do {
          temp_t = rtgamma_lower(0.5, 1.0 / eta1, 1.0);
          if (++tries > maxTries)
            Rcpp::stop("Qenetss: no valid truncated-gamma draw at step %d (eta1 = %g)", k + 1, eta1);
        } while (temp_t <= 1.0 || temp_t == R_PosInf);
        t(j) = temp_t;
      } else {
        double temp_lambda = 2.0 * eta1;
        double temp_nu = std::sqrt(temp_lambda / (2.0 * eta2 * beta(j) * beta(j)));
        double s;
        do {                                       // R redraws values <= 0, NA or Inf
          s = rinvGauss(temp_nu, temp_lambda);
          if (++tries > maxTries)
            Rcpp::stop("Qenetss: no valid inverse-Gaussian draw for t[%d] at step %d", j + 1, k + 1);
        } while (!(s > 0) || s == R_PosInf);
        t(j) = 1.0 / s + 1.0;
      }
    }
    tsample.row(k) = t.t();

    // ---- sample eta1: Metropolis-Hastings step (R lines 123-134) ----
    double temp_shape = p + c1;
    double temp_rate = arma::accu(t - 1.0) + d1;
    double temp_eta1 = R::rgamma(temp_shape, 1.0 / temp_rate);
    // log(pgamma(a)/pgamma(b)) computed as a difference of log upper tails (more stable)
    double logr = p * (R::pgamma(eta1, 0.5, 1.0, 0, 1) - R::pgamma(temp_eta1, 0.5, 1.0, 0, 1))
                + p / 2.0 * std::log(eta1 / temp_eta1) + p * (eta1 - temp_eta1);
    logr = std::min(logr, 0.0);
    if (std::log(R::runif(0.0, 1.0)) <= logr) eta1 = temp_eta1;
    eta1sample(k) = eta1;

    // ---- sample eta2 (R lines 136-140) ----
    double sumz = arma::accu(z);
    double shape2 = c2 + 0.5 * sumz;
    double rate2 = arma::accu(t / (t - 1.0) % arma::square(beta) % z) + d2;
    eta2 = R::rgamma(shape2, 1.0 / rate2);
    eta2sample(k) = eta2;

    // ---- sample pi (R lines 142-146) ----
    piVal = R::rbeta(r1 + p - sumz, u1 + sumz);
    pisample(k) = piVal;
  }

  // same names as the list returned by the R code (R lines 151-152)
  return Rcpp::List::create(Rcpp::Named("GS.beta") = betasample,
                            Rcpp::Named("t")       = tsample,
                            Rcpp::Named("tau")     = tausample,
                            Rcpp::Named("GS.b")    = Gammasamples,
                            Rcpp::Named("eta1")    = eta1sample,
                            Rcpp::Named("eta2")    = eta2sample,
                            Rcpp::Named("v")       = vsample,
                            Rcpp::Named("pi")      = pisample,
                            Rcpp::Named("GS.SS")   = SS);
}
