/*
 * File: dht.cc
 * Created Date: 2019-12-29
 * Author: Lei Pan
 * Contact: <panlei7@gmail.com>
 *
 * Last Modified: Sunday December 29th 2019 12:11:36 pm
 *
 * MIT License
 *
 * Copyright (c) 2019 Lei Pan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the 'Software'), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * -----
 * HISTORY:
 * Date      	 By	Comments
 * ----------	---
 * ----------------------------------------------------------
 */

#include "dht.hpp"

#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>

using namespace Eigen;
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using std::pow;

DiscreteHankelTransform::DiscreteHankelTransform(int order, double rmax, int nr)
    : order_(order), rmax_(rmax), nr_(nr), roots_(VectorXd::Zero(nr)) {
  for (int i = 0; i < nr_; ++i) {
    roots_(i) = cyl_bessel_j_zero(float(i + 1), order_);
  }
  for (int k = 0; k < nr_; ++k) {
    for (int m = 0; m < nr_; ++m) {
      tmatrix_(m, k) =
          2.0 / (roots_(nr_ - 1) * pow(cyl_bessel_j(order + 1, roots_(k)), 2)) *
          cyl_bessel_j(order, roots_(m) * roots_(k) / roots_(nr_ - 1));
    }
  }
}

DiscreteHankelTransform::~DiscreteHankelTransform() = default;

VectorXd DiscreteHankelTransform::r_sampling() {
  VectorXd r(nr_);
  for (int i = 0; i < nr_; ++i) {
    r(i) = roots_(i) * rmax_ / roots_(nr_ - 1);
  }
  return r;
}

VectorXd DiscreteHankelTransform::k_sampling() {
  VectorXd k(nr_);
  double kmax = roots_(nr_ - 1) / rmax_;
  for (int i = 0; i < nr_; ++i) {
    k(i) = roots_(i) * kmax / roots_(nr_ - 1);
  }
  return k;
}

VectorXd DiscreteHankelTransform::forward(const Ref<const VectorXd> &fr) {
  VectorXd fk(nr_);
  for (int m = 0; m < nr_; ++m) {
    fk(m) = pow(rmax_, 2) / roots_(nr_ - 1) * (tmatrix_.row(m) * fr).sum();
  }
  return fk;
}

VectorXd DiscreteHankelTransform::backward(const Ref<const VectorXd> &fk) {
  VectorXd fr(nr_);
  for (int k = 0; k < nr_; ++k) {
    fr(k) = roots_(nr_ - 1) / pow(rmax_, 2) * (tmatrix_.col(k) * fk).sum();
  }
  return fr;
}