#include <iostream>
#include <malloc.h>
#include <cstring>

#include <xmmintrin.h>

/* method provided by the code */
void sgemm_reference( int m, int n, float *A, float *C)
{
  #pragma omp parallel for
  for( int i = 0; i < m; i++ )
    for( int k = 0; k < n; k++ )
      for( int j = 0; j < m; j++ )
        C[i+j*m] += A[i+k*m] * A[j+k*m];
}

// Transpose a matrix (respecting the padding)
inline void Transpose(float * A, float * B, int m, int n, int bmPadded, int bnPadded)
{
  for(size_t i = 0; i < m; i++)
    for(size_t j = 0; j < n; j++)
      B[(i*bnPadded) + j] = A[i + (j*m)];
}

// Transpose a matrix
inline void Transpose(float * A, float * B, int m, int n)
{
  for(size_t i = 0; i < m; i++)
    for(size_t j = 0; j < n; j++)
      B[(i*n) + j] = A[i + (j*m)];
}

// Test
int main ()
{
  size_t m = 4;
  size_t n = 4;
  float  matrixA[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f, 16.0f};
  float *A  = (float *) matrixA;
  float *At = (float *) malloc (sizeof(float) * m * n);
  float  matrixC[m*m];
  float *C  = (float *) matrixC;

  // Print the source matrix
  std::cout << "Computing C = A * transpose(A)" << std::endl;
  std::cout << "A []:\t";
  for(size_t i = 0; i < m * m; i++)
    std::cout << matrixA[i] << "\t";
  std::cout << std::endl;

  // Compute and print the transpose matrix
  Transpose(A, At, m, n, m, n);
  std::cout << "A'[]:\t";
  for(size_t i = 0; i < m * m; i++)
    std::cout << At[i] << "\t";
  std::cout << std::endl;

  // Perform multiplication with the original method
  memset((void *) C, 0, sizeof(float) * m * m);
  sgemm_reference(m, n, A, C);
  std::cout << "C []:\t";
  for(size_t i = 0; i < m * m; i++)
    std::cout << C[i] << "\t";
  std::cout << std::endl << std::endl;

  // Zero the destination array
  memset((void *) C, 0, sizeof(float) * m * m);

  // Iterate through the columns of the matrix
  #pragma omp parallel for
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
          /*__m128 p0 = aColumn0 * _mm_load1_ps(At + (i*n) + (k+0));
          __m128 p1 = aColumn1 * _mm_load1_ps(At + (i*n) + (k+1));
          __m128 p2 = aColumn2 * _mm_load1_ps(At + (i*n) + (k+2));
          __m128 p3 = aColumn3 * _mm_load1_ps(At + (i*n) + (k+3));*/
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

  // Output resulting arrays
  std::cout << "Result using SSE optimized algorithm" << std::endl;
  std::cout << "C []:\t";
  for(size_t i = 0; i < m * m; i++)
    std::cout << C[i] << "\t";
  std::cout << std::endl << std::endl;

  return 0;
}
