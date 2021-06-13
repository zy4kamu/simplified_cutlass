#pragma once

#include "utils.h"

struct ThreadCalculator {
  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 64> & C,
    Array<float, 8> const & A,
    Array<float, 8> const & B) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < 8; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < 8; ++m) {
        C[m * 8 + n] += A[m] * B[n];
      }
    }
  }
};
