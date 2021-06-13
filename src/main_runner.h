#pragma once

#include "epilogue.h"
#include "params.h"
#include "threadblock_calculator.h"

union SharedStorage {
  ThreadblockSharedStorage main_loop;
  AlignedBuffer<float, 16 * 128> epilogue;
};

__global__
void Kernel(Params params) {
  extern __shared__ int SharedStorageBase[];
  SharedStorage *shared_storage = reinterpret_cast<SharedStorage *>(SharedStorageBase);

  int problem_size_k = params.problem_size[2];
  int gemm_k_iterations = (problem_size_k + 7) / 8;

  ThreadblockCalculator mma(params, shared_storage->main_loop);
  Array<float, 64> accumulators;
  accumulators.clear();
  mma(gemm_k_iterations, accumulators, accumulators);

  Epilogue epilogue(params, shared_storage->epilogue);
  epilogue(accumulators);
}

struct MainRunner {
  bool operator()(Params const &args) {
    dim3 grid = dim3((args.problem_size[0] + 127) / 128,
                     (args.problem_size[1] + 127) / 128,
                     1);
    dim3 block(32 * 8, 1, 1);

    int smem_size = int(sizeof(SharedStorage));
    if (smem_size >= (48 << 10)) {
      cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaFuncSetAttribute(Kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    }
    Kernel<<<grid, block, smem_size>>>(args);
    return cudaGetLastError() == cudaSuccess;
  }
};
