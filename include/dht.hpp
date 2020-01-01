/*
 * File: dht.hpp
 * Created Date: 2019-12-29
 * Author: Lei Pan
 * Contact: <panlei7@gmail.com>
 *
 * Last Modified: Sunday December 29th 2019 12:00:12 pm
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

#ifndef DHT_DHT_H_
#define DHT_DHT_H_

#include <Eigen/Dense>

class DiscreteHankelTransform {
public:
  DiscreteHankelTransform(int order, int nr);
  ~DiscreteHankelTransform();

  Eigen::VectorXd r_sampling(double rmax);
  Eigen::VectorXd k_sampling(double rmax); // k is rho in the paper

  Eigen::VectorXd forward(const Eigen::Ref<const Eigen::VectorXd> &fr);
  Eigen::VectorXd backward(const Eigen::Ref<const Eigen::VectorXd> &fk);

  Eigen::VectorXd shift(const Eigen::Ref<const Eigen::VectorXd> &raw, int m);

  Eigen::MatrixXd tmatrix() const;

private:
  int order_;
  double rmax_;
  int nr_;

  Eigen::VectorXd roots_;
  Eigen::MatrixXd tmatrix_;
};

#endif