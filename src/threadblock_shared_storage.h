#pragma once

#include "utils.h"

class ThreadblockSharedStorage {
 public:
  AlignedBuffer<float, 8 * 128> operand_A;
  AlignedBuffer<float, 8 * 128> operand_B;
};
