#pragma once

#include "utils.h"

class ThreadBlockSharedMemoryBufferA {
public:
  CUTLASS_DEVICE
  ThreadBlockSharedMemoryBufferA(float* data):
    pointer_(data + threadIdx.x / 8 + 128 * (threadIdx.x % 8))
  {}

  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Array<float, 4> const &frag) {
    float const *frag_ptr = reinterpret_cast<const float*>(&frag);
    float *access_ptr = reinterpret_cast<float *>(pointer_);
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < 4; ++c) {
      access_ptr[c * 32] = frag_ptr[c];
    }
  }

private:
  float *pointer_;
};
