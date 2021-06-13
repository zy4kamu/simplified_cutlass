#pragma once

#include "utils.h"

class ThreadBlockIteratorDramToRegistersA {
 private:
  float* pointer_;
  int stride_;

 public:
  CUTLASS_HOST_DEVICE
  ThreadBlockIteratorDramToRegistersA(int stride, float* pointer, int matrix_side_size)
      : stride_(stride),
        pointer_(pointer) {
    pointer_ += threadIdx.x % 8 + (blockIdx.x * 128 + threadIdx.x / 8) * stride_;
  }

  CUTLASS_DEVICE
  ThreadBlockIteratorDramToRegistersA &operator++() {
    pointer_ += 8;
    return *this;
  }

  CUTLASS_DEVICE
  void load(Array<float, 4> &frag) {
    float *frag_ptr = reinterpret_cast<float *>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < 4; ++idx) {
      frag_ptr[idx] = pointer_[idx * stride_ * 32];
    }
  }
};
