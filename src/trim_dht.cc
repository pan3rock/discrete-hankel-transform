#include "trim_dht.hpp"
#include "dht.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <iostream>
#include <iterator>

using namespace Eigen;
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using std::pow;

TrimDHT::TrimDHT(int order, int nr, double rmax, int nexp)
    : DiscreteHankelTransform(order, nr * pow(2, nexp)),
      shift_matrix_(nr_, nr_) {
  // nr_ = nr * pow(2, nexp)
  rmax_extend_ = cyl_bessel_j_zero(float(order), nr_ + 1) /
                 cyl_bessel_j_zero(float(order), nr + 1) * rmax;
  shift_matrix_ = tmatrix_.leftCols(nr) * tmatrix_.topRows(nr);
}

TrimDHT::~TrimDHT() = default;

VectorXd TrimDHT::perform(const Ref<const VectorXd> &src) {
  VectorXd ret;
  ret = shift_matrix_ * src;
  return ret;
}

VectorXd TrimDHT::r_sampling() {
  return DiscreteHankelTransform::r_sampling(rmax_extend_);
}

VectorXd TrimDHT::k_sampling() {
  return DiscreteHankelTransform::k_sampling(rmax_extend_);
}

int TrimDHT::get_nr() const { return nr_; }