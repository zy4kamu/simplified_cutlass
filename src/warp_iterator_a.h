#pragma once

#include "utils.h"

class WarpIteratorA {
public:
  CUTLASS_HOST_DEVICE
  WarpIteratorA(float* data) {
    int shift = (threadIdx.x / 32) % 4;
    int row_major = (threadIdx.x % 32) / 16;
    int row_minor = threadIdx.x % 2;
    ptr_ = (Array<float, 4>*)data + shift * 8 + row_major * 2 + row_minor;
  }

  CUTLASS_HOST_DEVICE
  void reset() {
    ptr_ -= 2 * 128;
  }

  CUTLASS_HOST_DEVICE
  WarpIteratorA & operator++() {
    ptr_ += 32;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void load(Array<float, 8> &frag) const {
    Array<float, 4> *dst_ptr = reinterpret_cast<Array<float, 4> *>(&frag);
    dst_ptr[0] = ptr_[0];
    dst_ptr[1] = ptr_[4];
  }
private:
  Array<float, 4>* ptr_;
};
