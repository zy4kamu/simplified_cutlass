#pragma once

#include "utils.h"

class EpilogueDramIterator {
private:
  int stride_;
  float *ptr_;
  int state_ = 0;

public:

  CUTLASS_DEVICE
  EpilogueDramIterator(
    int stride,
    float *pointer
  ): 
    stride_(stride)
  {
    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    ptr_ = pointer + (blockIdx.x * stride_ + blockIdx.y) * 128 +
           (32 * (warp_idx / 2) + 4 * (warp_idx % 2)) * stride_ +
           lane_idx;
  }

  CUTLASS_DEVICE
  void store(Array<float, 8> const &frag) {
    float const *frag_ptr = reinterpret_cast<float const *>(&frag);
    for (int column = 0; column < 4; ++column) {
      ptr_[column * 32] = frag_ptr[column];
      ptr_[column * 32 + stride_ * 8] = frag_ptr[4 + column];
    }
  }

  CUTLASS_HOST_DEVICE
  EpilogueDramIterator &operator++() {
    ptr_ += (++state_ == 4) ? stride_ * 13 : stride_;
    return *this;
  }
};
