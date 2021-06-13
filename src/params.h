#pragma once

#include "utils.h"

struct Params {
  Array<int, 3> problem_size;
  float* A, *B, *C, *D;
  int strideA, strideB, strideC, strideD;

  CUTLASS_HOST_DEVICE
  Params(
    Array<int, 3> const & problem_size,
    float* A, float* B, float* C, float* D,
    int strideA, int strideB, int strideC, int strideD):
    problem_size(problem_size),
    A(A), B(B), C(C), D(D),
    strideA(strideA),
    strideB(strideB),
    strideC(strideC),
    strideD(strideD)
  {
  }
};
