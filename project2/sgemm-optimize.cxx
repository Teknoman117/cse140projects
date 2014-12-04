#include <iostream>
#include <cstdlib>
#include <cstring>

#include <xmmintrin.h>

// Computes C = A * transpose(A).  m and n must be multiples of 4
inline void atimestransposea( int m, int n, float *A, float *C )
{
  // Iterate through the columns of the matrix
  //#pragma omp parallel for
  for(size_t i = 0; i < m; i++)
  {
    // Iterate through the rows of the matrix
    for(size_t j = 0; j < m; j += 4)
    {
      // Summation register for these elements
      __m128 result = _mm_set1_ps(0.0f);

      // Iterate over the elements in a row
      for(size_t k = 0; k < n; k += 4)
      {
        // Load columns of left matrix into sse registers
        __m128 aColumn0 = _mm_load_ps(A + ((k+0)*m) + j);
        __m128 aColumn1 = _mm_load_ps(A + ((k+1)*m) + j);
        __m128 aColumn2 = _mm_load_ps(A + ((k+2)*m) + j);
        __m128 aColumn3 = _mm_load_ps(A + ((k+3)*m) + j);

        // Multiply each column by the cooresponding entry in the right matrix
        __m128 p0 = aColumn0 * _mm_load1_ps(A + ((k+0)*m) + i);
        __m128 p1 = aColumn1 * _mm_load1_ps(A + ((k+1)*m) + i);
        __m128 p2 = aColumn2 * _mm_load1_ps(A + ((k+2)*m) + i);
        __m128 p3 = aColumn3 * _mm_load1_ps(A + ((k+3)*m) + i);

        // Store the summations
        result += p0;
        result += p1;
        result += p2;
        result += p3;
      }

      // Store result
      _mm_store_ps(C + (i*m) + j, result);
    }
  }
}

// Multiply A matrix by its transpose
extern "C" void sgemm( int m, int n, float *A, float *C )
{
    // The matrix does not have favorable dimensions
    if((m % 4) || (n % 4))
    {
        // Recompute boundaries of the matrix (align to 4, for sse)
        int mPadded = (m & ~0x03) + ((m & 0x03) ? 4 : 0);
        int nPadded = (n & ~0x03) + ((n & 0x03) ? 4 : 0);

        // Allocate new, padded matrices
        float *Apadded = (float *) malloc (sizeof(float) * mPadded * nPadded);
        float *Cpadded = (float *) malloc (sizeof(float) * mPadded * mPadded);
        memset((void *) Apadded, 0, sizeof(float) * mPadded * nPadded);
        memset((void *) Cpadded, 0, sizeof(float) * mPadded * mPadded);

        // Perform copy of A into padded matrix (optimized for column major matrices, hopefully we get accelerated memcpy)
        for(int j = 0; j < n; j++)
            memcpy((void *) (Apadded + (mPadded*j)), (void *) (A + (m*j)), sizeof(float) * m);

        // Perform multiplication
        atimestransposea(mPadded, nPadded, Apadded, Cpadded);

        // Perform a copy of Cpadded into C matrix (optimized for column major matrices)
        for(int j = 0; j < m; j++)
            memcpy((void *) (C + (m*j)), (void *) (Cpadded + (mPadded*j)), sizeof(float) * m);

        // Cleanup
        free(Apadded);
        free(Cpadded);
    }

    // Otherwise, this is an optimal case where padding is not required
    else
    {
        atimestransposea(m, n, A, C);
    }
}
