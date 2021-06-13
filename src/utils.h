#pragma once

#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#define CUTLASS_PRAGMA_UNROLL _Pragma("unroll")

template <typename T, int N>
class Array {
private:
  T storage[N];

public:

  CUTLASS_HOST_DEVICE
  Array() { }

  CUTLASS_HOST_DEVICE
  Array(Array const &x) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      storage[i] = x.storage[i];
    }
  }

  CUTLASS_HOST_DEVICE
  void clear() {
    fill(T(0));
  }

  CUTLASS_HOST_DEVICE
  T& operator[](size_t pos) {
    return storage[pos];
  }

  CUTLASS_HOST_DEVICE
  const T& operator[](size_t pos) const {
    return storage[pos];
  }

  CUTLASS_HOST_DEVICE
  void fill(T const &value) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      storage[i] = value;
    }
  }
};

template <typename T, int N>
struct AlignedBuffer {
  alignas(16) uint8_t storage[sizeof(T) * N];
  CUTLASS_HOST_DEVICE
  float* data() {
    return reinterpret_cast<float*>(storage);
  }
};
