#pragma once

#include "thread_calculator.h"
#include "threadblock_iterator_dram_to_registers_a.h"
#include "threadblock_iterator_dram_to_registers_b.h"
#include "threadblock_shared_memory_buffer_a.h"
#include "threadblock_shared_memory_buffer_b.h"
#include "threadblock_shared_storage.h"
#include "warp_iterator_a.h"
#include "warp_iterator_b.h"

class ThreadblockCalculator {
protected:
  WarpIteratorA warp_tile_iterator_A_;
  WarpIteratorB warp_tile_iterator_B_;
  ThreadBlockSharedMemoryBufferA smem_iterator_A_;
  ThreadBlockSharedMemoryBufferB smem_iterator_B_;
  ThreadBlockIteratorDramToRegistersA iterator_dram_to_registers_A;
  ThreadBlockIteratorDramToRegistersB iterator_dram_to_registers_B;

public:

  CUTLASS_DEVICE
  ThreadblockCalculator(
    Params const &params,
    ThreadblockSharedStorage &shared_storage
  ):
    warp_tile_iterator_A_(shared_storage.operand_A.data()),
    warp_tile_iterator_B_(shared_storage.operand_B.data()),
    smem_iterator_A_(shared_storage.operand_A.data()),
    smem_iterator_B_(shared_storage.operand_B.data()),
    iterator_dram_to_registers_A(params.strideA, params.A, params.problem_size[0]),
    iterator_dram_to_registers_B(params.strideB, params.B, params.problem_size[1]) {
  }

  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,
    Array<float, 64> &accum,
    Array<float, 64> const &src_accum) {

    accum = src_accum;
    Array<float, 4> regisers_file_A;
    Array<float, 4> regisers_file_B;
    Array<float, 8> warp_frag_A;
    Array<float, 8> warp_frag_B;
    ThreadCalculator thread_calculator;

    CUTLASS_PRAGMA_UNROLL
    for (size_t t = 0; t < gemm_k_iterations; ++t) {
      iterator_dram_to_registers_A.load(regisers_file_A);
      iterator_dram_to_registers_B.load(regisers_file_B);
      ++iterator_dram_to_registers_A;
      ++iterator_dram_to_registers_B;
      this->smem_iterator_A_.store_with_pointer_offset(regisers_file_A);
      this->smem_iterator_B_.store_with_pointer_offset(regisers_file_B);
      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (size_t i = 0; i < 8; ++i) {
        this->warp_tile_iterator_A_.load(warp_frag_A);
        this->warp_tile_iterator_B_.load(warp_frag_B);
        thread_calculator(accum, warp_frag_A, warp_frag_B);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
      }

      this->warp_tile_iterator_A_.reset();
      this->warp_tile_iterator_B_.reset();
      __syncthreads();
    }
  }
};
