/*
 * File: pybind.cc
 * Created Date: 2019-12-29
 * Author: Lei Pan
 * Contact: <panlei7@gmail.com>
 *
 * Last Modified: Sunday December 29th 2019 1:42:01 pm
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
#include "trim_dht.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(dhtcxx, m) {
  py::class_<DiscreteHankelTransform>(m, "DiscreteHankelTransform")
      .def(py::init<int, int>())
      .def("r_sampling", &DiscreteHankelTransform::r_sampling)
      .def("k_sampling", &DiscreteHankelTransform::k_sampling)
      .def("forward", &DiscreteHankelTransform::forward)
      .def("backward", &DiscreteHankelTransform::backward)
      .def("shift", &DiscreteHankelTransform::shift)
      .def("tmatrix", &DiscreteHankelTransform::tmatrix);
  py::class_<TrimDHT>(m, "TrimDHT")
      .def(py::init<int, int, double, int>())
      .def("r_sampling", &TrimDHT::r_sampling)
      .def("k_sampling", &TrimDHT::k_sampling)
      .def("get_nr", &TrimDHT::get_nr)
      .def("perform", &TrimDHT::perform);
}
