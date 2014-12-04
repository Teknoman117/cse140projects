#include <iostream>
#include <malloc.h>
#include <cstring>

/* corrected method */
void sgemm_fixed( int m, int n, float *A, float *C )
{
  #pragma omp parallel for
  for( int i = 0; i < n; i++)
    for( int k = 0; k < m; k++)
      for( int j = 0; j < n; j++)
        C[j+(i*n)] += A[j+(k*n)] * A[i+(k*n)];
}

/* method provided by the code */
void sgemm_reference( int m, int n, float *A, float *C)
{
  #pragma omp parallel for
  for( int i = 0; i < n; i++ )
    for( int k = 0; k < m; k++ )
      for( int j = 0; j < n; j++ )
        C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
}

// Test
int main ()
{
  // Test matrix (3x2) in column major
  // [{1.0, 4.0},
  //  {2.0, 5.0},
  //  {3.0, 6.0}]
  float matrixA[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  size_t n = 3;
  size_t m = 2;

  // Allocate destination matrix
  float *C = (float *) malloc (sizeof(float) * n * n);

  // Print the source mmatrix
  std::cout << "A[]:\t";
  for(size_t i = 0; i < sizeof(matrixA) / sizeof(float); i++)
    std::cout << matrixA[i] << "\t";
  std::cout << std::endl << std::endl;

  std::cout << "Operation being performed ( C = A * transpose(A) )" << std::endl;
  std::cout << " [{1.0, 4.0},                        [{17.0, 22.0, 27.0}," << std::endl;
  std::cout << "  {2.0, 5.0}, * [{1.0, 2.0, 3.0}, =   {22.0, 29.0, 36.0}," << std::endl;
  std::cout << "  {3.0, 6.0}]    {4.0, 5.0, 6.0}]     {27.0, 36.0, 45.0}]" << std::endl;
  std::cout << std::endl;
  std::cout << " Expected result (in column major) = 17,22,27,22,29,36,27,36,45" << std::endl;
  std::cout << std::endl;

  // Perform multiplication with the original method
  memset((void *) C, 0, sizeof(float) * n * n);
  sgemm_reference(m, n, (float *) matrixA, C);
  std::cout << "C (provided) []:\t";
  for(size_t i = 0; i < n * n; i++)
    std::cout << C[i] << "\t";
  std::cout << std::endl;

  // Perform multication with the corrected method
  memset((void *) C, 0, sizeof(float) * n * n);
  sgemm_fixed(m, n, (float *) matrixA, C);
  std::cout << "C (corrected) []:\t";
  for(size_t i = 0; i < n * n; i++)
    std::cout << C[i] << "\t";
  std::cout << std::endl;

  return 0;
}
