#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one  = _pp_vset_int(1);
  __pp_vec_float clampValue = _pp_vset_float(9.999999f);

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // 計算還剩下多少元素
    int remain = N - i;
    int maskWidth = remain < VECTOR_WIDTH ? remain : VECTOR_WIDTH;
    __pp_mask mask = _pp_init_ones(maskWidth);

    // 載入資料
    __pp_vec_float x, result;
    __pp_vec_int y;
    _pp_vload_float(x, values + i, mask);
    _pp_vload_int(y, exponents + i, mask);

    // case1: exponent == 0 → result = 1
    __pp_mask maskEq0;
    _pp_veq_int(maskEq0, y, zero, mask);
    _pp_vset_float(result, 1.f, maskEq0);

    // case2: exponent != 0 → result = x, y = y - 1
    __pp_mask maskNeq0 = _pp_mask_not(maskEq0);
    _pp_vmove_float(result, x, maskNeq0);
    _pp_vsub_int(y, y, one, maskNeq0);

    // while (y > 0) result *= x;
    __pp_mask maskGt0;
    _pp_vgt_int(maskGt0, y, zero, mask);
    while (_pp_cntbits(maskGt0) > 0) {
      _pp_vmult_float(result, result, x, maskGt0);
      _pp_vsub_int(y, y, one, maskGt0);
      _pp_vgt_int(maskGt0, y, zero, mask);
    }

    // clamp result > 9.999999f
    __pp_mask maskClamp;
    _pp_vgt_float(maskClamp, result, clampValue, mask);
    _pp_vmove_float(result, clampValue, maskClamp);

    // store 回 output
    _pp_vstore_float(output + i, result, mask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float vecSum = _pp_vset_float(0.f);
  __pp_vec_float vec;
  __pp_mask maskAll = _pp_init_ones();

  // 向量加總
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    _pp_vload_float(vec, values + i, maskAll);
    _pp_vadd_float(vecSum, vecSum, vec, maskAll);
  }

  // 向量歸約成 scalar
  int shift = VECTOR_WIDTH;
  while (shift > 1) {
    _pp_hadd_float(vecSum, vecSum);
    _pp_interleave_float(vecSum, vecSum);
    shift >>= 1;
  }

  return vecSum.value[0];
}


