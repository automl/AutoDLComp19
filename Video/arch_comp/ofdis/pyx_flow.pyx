# distutils: language = c++
# distutils: sources = run_dense.cpp
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from opencv_mat cimport *
# Author: Deepak Pathak (c) 2016


## define c extension extracted from from msrc.h for add(int a,int b)
cdef extern from "run_dense.h":
    Mat optical_flow(Mat a, Mat b);

cdef extern from "opencv2/opencv.hpp": 
    cdef int CV_WINDOW_AUTOSIZE       
    cdef int CV_8UC3
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_64FC1


cdef Mat np2Mat3D(np.ndarray ary):
    assert ary.ndim==3 and ary.shape[2]==3, "ASSERT::3channel RGB only!!"
    ary = np.dstack((ary[...,2], ary[...,1], ary[...,0])) #RGB -> BGR

    cdef np.ndarray[np.uint8_t, ndim=3, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.uint8)
    cdef unsigned int* im_buff = <unsigned int*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_8UC3)
    memcpy(m.data, im_buff, r*c*3)
    return m


cdef Mat np2Mat2D(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"

    cdef np.ndarray[np.uint8_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.uint8)
    cdef unsigned int* im_buff = <unsigned int*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_8UC1)
    memcpy(m.data, im_buff, r*c)
    return m


cdef Mat np2Mat(np.ndarray ary):
    if ary.ndim == 2:
        return np2Mat2D(ary)
    elif ary.ndim == 3:
        return np2Mat3D(ary)


cdef object Mat2np(Mat m):
    # Create buffer to transfer data from m.data
    cdef Py_buffer buf_info
    # Define the size / len of data
    cdef size_t len = m.rows*m.cols*m.channels()*sizeof(CV_32FC1)
    # Fill buffer
    PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)
    # Get Pyobject from buffer data
    Pydata  = PyMemoryView_FromBuffer(&buf_info)

    # Create ndarray with data
    shape_array = (m.rows, m.cols, m.channels())
    ary = np.ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=np.float32)
    result = np.copy(ary)
    return result


## Define the python wrapper that pass calls to the c++ function add(a,b)
def optical_fn(a,b):
    cdef Mat m
    a1 = np2Mat(a)
    b1 = np2Mat(b)
    m = optical_flow(a1,b1)
    return Mat2np(m)





