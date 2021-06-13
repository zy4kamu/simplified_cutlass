#pragma once

#include "epilogue_dram_iterator.h"
#include "epilogue_shared_memory_buffer.h"
#include "epilogue_warp_iterator.h"
#include "params.h"

class Epilogue {
public:
  CUTLASS_DEVICE
  Epilogue(Params params, AlignedBuffer<float, 16 * 128>& shared_storage):
    warp_iterator_(shared_storage.data()),
    shared_memory_buffer_(shared_storage.data()),
    destination_iterator(params.strideD, params.D)
  {
  }

  CUTLASS_DEVICE
  void operator()(Array<float, 64> const &accumulators) {
    Array<float, 8> const *acc = reinterpret_cast<Array<float, 8> const *>(&accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < 8; ++iter) {
      __syncthreads();
      this->warp_iterator_.store(acc[iter]);
      __syncthreads();

      Array<float, 8> aligned_accum_fragment;
      shared_memory_buffer_.load(aligned_accum_fragment);

      destination_iterator.store(aligned_accum_fragment);
      ++destination_iterator;
    }
  }

private:
  EpilogueWarpIterator warp_iterator_;
  EpilogueSharedMemoryBuffer shared_memory_buffer_;
  EpilogueDramIterator destination_iterator;
};
