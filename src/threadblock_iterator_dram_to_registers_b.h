#pragma once

#include "utils.h"

class ThreadBlockIteratorDramToRegistersB {
 private:
  float* pointer_;
  int stride_;

 public:
  CUTLASS_HOST_DEVICE
  ThreadBlockIteratorDramToRegistersB(int stride, float* pointer, int matrix_side_size)
      : stride_(stride),
        pointer_(pointer) {
    pointer_ += blockIdx.y * 128 + threadIdx.x % 128 + (threadIdx.x / 128) * stride_;
  }

  CUTLASS_DEVICE
  ThreadBlockIteratorDramToRegistersB &operator++() {
    pointer_ += 8 * stride_;
    return *this;
  }

  CUTLASS_DEVICE
  void load(Array<float, 4> &frag) {
    float *frag_ptr = reinterpret_cast<float *>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < 4; ++idx) {
      frag_ptr[idx] = pointer_[idx * stride_ * 2];
    }
  }
};
