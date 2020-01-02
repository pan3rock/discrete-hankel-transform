/*
 * File: trim_dht.hpp
 * Created Date: 2020-01-01
 * Author: Lei Pan
 * Contact: <panlei7@gmail.com>
 *
 * Last Modified: Wednesday January 1st 2020 11:27:12 am
 *
 * MIT License
 *
 * Copyright (c) 2020 Lei Pan
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

#ifndef DHT_TRIM_DHT_H_
#define DHT_TRIM_DHT_H_

#include "dht.hpp"
#include <Eigen/Dense>

class TrimDHT : public DiscreteHankelTransform {
public:
  TrimDHT(int order, int nr, double rmax, int nexp);
  ~TrimDHT();

  Eigen::VectorXd perform(const Eigen::Ref<const Eigen::VectorXd> &src);

  Eigen::VectorXd r_sampling();
  Eigen::VectorXd k_sampling();

private:
  Eigen::MatrixXd shift_matrix_;
  double rmax_extend_;
};

#endif