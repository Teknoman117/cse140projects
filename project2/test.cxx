#include <iostream>
#include <malloc.h>
#include <cstring>

#include <xmmintrin.h>

/* method provided by the code */
void sgemm_reference( int m, int n, float *A, float *C)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < m; j++)
            for(size_t k = 0; k < n; k++)
                 C[(i*m)+j] += A[(j*n) + k] * A[(i*m) + k];
}

// Computes C = A * transpose(A).  m and n must be multiples of 4. A is row major, C is column major
void atimestransposea( int m, int n, float *A, float *C )
{
    // Iterate through the columns of the matrix
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
                // Load r-matrix column
                __m128 aColumn = _mm_load_ps(A + (i*n) + k);

                // Load l-matrix rows
                __m128 aRow0 = _mm_load_ps(A + ((j+0)*n) + k) * aColumn;
                __m128 aRow1 = _mm_load_ps(A + ((j+1)*n) + k) * aColumn;
                __m128 aRow2 = _mm_load_ps(A + ((j+2)*n) + k) * aColumn;
                __m128 aRow3 = _mm_load_ps(A + ((j+3)*n) + k) * aColumn;

                // Transpose the row so that we can use vector add
                _MM_TRANSPOSE4_PS(aRow0, aRow1, aRow2, aRow3);

                // Store the summations
                result += aRow0;
                result += aRow1;
                result += aRow2;
                result += aRow3;
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

        // Perform a copy of A into padded matrix (optimized for row major matrices)
        for(int j = 0; j < m; j++)
            memcpy((void *) (Apadded + (nPadded*j)), (void *) (A + (n*j)), sizeof(float) * n);

        // Perform multiplication
        atimestransposea(mPadded, nPadded, Apadded, Cpadded);

        // Perform a copy of Cpadded into C matrix (optimized for column major matrices)
        for(int j = 0; j < m; j++)
            memcpy((void *) (C + (m*j)), (void *) (Cpadded + (mPadded*j)), sizeof(float) * m);

        // Cleanup
        free(Cpadded);
        free(Apadded);
    }

    // Otherwise, this is an optimal case where padding is not required
    else
    {
        atimestransposea(m, n, A, C);
    }
}

// Test
int main ()
{
  size_t m = 8;
  size_t n = 4;
  float  matrixA[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  float  matrixC[m*m];
  float *A  = (float *) matrixA;
  float *C  = (float *) matrixC;

  // Print the source matrix
  std::cout << "Computing C = A * transpose(A)" << std::endl;
  std::cout << "A[]:\t";
  for(size_t i = 0; i < m * n; i++)
    std::cout << matrixA[i] << "\t";
  std::cout << std::endl;

  // Perform multiplication with the original method
  memset((void *) C, 0, sizeof(float) * m * m);
  sgemm_reference(m, n, A, C);
  std::cout << "C[]:\t";
  for(size_t i = 0; i < m * m; i++)
    std::cout << C[i] << "\t";
  std::cout << std::endl << std::endl;

  // Zero the destination array
  memset((void *) C, 0, sizeof(float) * m * m);

  // Iterate through the columns of the matrix
  sgemm(m,n,A,C);

  // Output resulting arrays
  std::cout << "Result using SSE optimized algorithm" << std::endl;
  std::cout << "C[]:\t";
  for(size_t i = 0; i < m * m; i++)
    std::cout << C[i] << "\t";
  std::cout << std::endl << std::endl;

  return 0;
}
