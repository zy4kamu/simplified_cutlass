#pragma once

#include "utils.h"

class ThreadBlockSharedMemoryBufferB {
public:
  CUTLASS_DEVICE
  ThreadBlockSharedMemoryBufferB(float* data) {
    pointer_ = data + threadIdx.x;
  }

  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Array<float, 4> const &frag) {
    float const *frag_ptr = reinterpret_cast<float const*>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < 4; ++s) {
      pointer_[s * 128 * 2] = frag_ptr[s];
    }
  }
private:
  float *pointer_;
};
