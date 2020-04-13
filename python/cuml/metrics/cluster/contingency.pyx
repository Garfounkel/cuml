#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuml.common.handle cimport cumlHandle
from libc.stdint cimport uintptr_t

from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs
import cuml.common.handle
import numpy as np

from cuml.common import CumlArray


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    void contingencyMatrix(const cumlHandle &handle,
                           int *out_mat,
                           const int *y, const int *y_hat,
                           const int n, const int lower_class_range,
                           const int upper_class_range) except +


def contingency_matrix(labels_true, labels_pred, handle=None):
    """
    TODO: Add documentation
    """
    cdef uintptr_t ground_truth_ptr
    cdef uintptr_t preds_ptr

    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle *handle_ = <cumlHandle*> <size_t> handle.getHandle()

    # from cuml.prims.label.classlabels import make_monotonic
    # labels_true, true_id = make_monotonic(labels_true)
    # print(labels_true, true_id)
    # labels_pred, pred_id = make_monotonic(labels_pred, true_id)
    # print(labels_pred, pred_id)

    (ground_truth_ptr, preds_ptr,
     n_rows,
     lower_class_range, upper_class_range) = prepare_cluster_metric_inputs(
        labels_true,
        labels_pred
    )

    num_classes = upper_class_range - lower_class_range + 1
    out_mat = CumlArray.zeros(shape=(num_classes, num_classes),
                              dtype=np.int32, order='C')
    print(num_classes)
    cdef uintptr_t out_ptr = out_mat.ptr

    contingencyMatrix(handle_[0],
                      <int*> out_ptr,
                      <int*> ground_truth_ptr,
                      <int*> preds_ptr,
                      <int> n_rows,
                      <int> lower_class_range,
                      <int> upper_class_range)

    return out_mat
