#include <iostream>
#include <malloc.h>
#include <cstring>
#include <sstream>

#include <xmmintrin.h>

// Index into an MxN matrix.  r = row #, c = column #
#define ROW_MAJOR(r,c,M,N)    ((r)*(N))+(c)
#define COLUMN_MAJOR(r,c,M,N) ((c)*(M))+(r)
#define MIN(a,b) (a<b)?a:b

/* print row major matrix */
void print_rowmajor(float *A, int m, int n, std::string name)
{
    std::cout << name;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            std::cout << "\t" << A[ROW_MAJOR(i,j,m,n)];
        std::cout << std::endl;
    }
}

/* print column major matrix */
void print_columnmajor(float *A, int m, int n, std::string name)
{
    std::cout << name;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            std::cout << "\t" << A[COLUMN_MAJOR(i,j,m,n)];
        std::cout << std::endl;
    }
}

// Computes product of 2 4x4 matrices (C += A * B)
//   A -> row major matrix
//   B -> column major matrix
//   C -> column major matrix
void block_multiply_accumulate( float *A, float *B, float *C )
{
    // Intermediates for summation
    __m128 intermediate0, intermediate1, intermediate2, intermediate3;

    // Load all the rows of the left matrix
    __m128 row0 = _mm_load_ps(A + 0);
    __m128 row1 = _mm_load_ps(A + 4);
    __m128 row2 = _mm_load_ps(A + 8);
    __m128 row3 = _mm_load_ps(A + 12);

    // Compute the sums for the first column
    __m128 column0      = _mm_load_ps(B + 0);
    __m128 destination0 = _mm_load_ps(C + 0);
    intermediate0 = row0 * column0;
    intermediate1 = row1 * column0;
    intermediate2 = row2 * column0;
    intermediate3 = row3 * column0;
    _MM_TRANSPOSE4_PS(intermediate0, intermediate1, intermediate2, intermediate3);
    destination0 += intermediate0 + intermediate1 + intermediate2 + intermediate3;
    _mm_store_ps(C + 0, destination0);

    // Compute the sums for the second column
    __m128 column1      = _mm_load_ps(B + 4);
    __m128 destination1 = _mm_load_ps(C + 4);
    intermediate0 = row0 * column1;
    intermediate1 = row1 * column1;
    intermediate2 = row2 * column1;
    intermediate3 = row3 * column1;
    _MM_TRANSPOSE4_PS(intermediate0, intermediate1, intermediate2, intermediate3);
    destination1 += intermediate0 + intermediate1 + intermediate2 + intermediate3;
    _mm_store_ps(C + 4, destination1);

    // Compute the sums for the third column
    __m128 column2      = _mm_load_ps(B + 8);
    __m128 destination2 = _mm_load_ps(C + 8);
    intermediate0 = row0 * column2;
    intermediate1 = row1 * column2;
    intermediate2 = row2 * column2;
    intermediate3 = row3 * column2;
    _MM_TRANSPOSE4_PS(intermediate0, intermediate1, intermediate2, intermediate3);
    destination2 += intermediate0 + intermediate1 + intermediate2 + intermediate3;
    _mm_store_ps(C + 8, destination2);

    // Compute the sums for the fourth column
    __m128 column3      = _mm_load_ps(B + 12);
    __m128 destination3 = _mm_load_ps(C + 12);
    intermediate0 = row0 * column3;
    intermediate1 = row1 * column3;
    intermediate2 = row2 * column3;
    intermediate3 = row3 * column3;
    _MM_TRANSPOSE4_PS(intermediate0, intermediate1, intermediate2, intermediate3);
    destination3 += intermediate0 + intermediate1 + intermediate2 + intermediate3;
    _mm_store_ps(C + 12, destination3);
}

// Multiply A matrix by its transpose
extern "C" void sgemm( int m, int n, float *A, float *C )
{
    // Compute dimensions of "matrix of blocks"
    int mBlocks = (m / 4) + ((m & 0x03) ? 1 : 0);
    int nBlocks = (n / 4) + ((n & 0x03) ? 1 : 0);

    // might be needed
    int mPadded = (m & ~0x03) + ((m & 0x03) ? 4 : 0);
    int nPadded = (n & ~0x03) + ((n & 0x03) ? 4 : 0);

    // Allocate a pointer arrays to perform block allocation in
    float **source_blocks = (float **) calloc (mBlocks * nBlocks, sizeof(float *));
    float **transpose_blocks = (float **) calloc (mBlocks * nBlocks, sizeof(float *));
    float **destination_blocks = (float **) calloc (mBlocks * mBlocks, sizeof(float *));

    // Perform destination matrix allocation
    for(int i = 0; i < mBlocks * mBlocks; i++)
        destination_blocks[i] = (float *) calloc (16, sizeof(float));

    // Perform source matrix allocation
    //std::cout << "------- SOURCE MATRIX ---------" << std::endl;
    for(int i = 0; i < mBlocks; i++)   // i = row
    {
        for(int j = 0; j < nBlocks; j++) // j = column
        {
            // Allocate memory for the block
            float *block = (float *) calloc (16, sizeof(float));
            source_blocks[ROW_MAJOR(i,j,mBlocks,nBlocks)] = block;

            // Compute the dimensions of this block
            int mLocalBlock = MIN(m - (4 * i), 4);
            int nLocalBlock = MIN(n - (4 * j), 4);

            // Copy and transpose, ii = row, jj = column
            for(int ii = 0; ii < mLocalBlock; ii++)
                for(int jj = 0; jj < nLocalBlock; jj++)
                {
                    int bIdx = ROW_MAJOR(ii,jj,4,4);
                    int sIdx = COLUMN_MAJOR((i*4)+ii,(j*4)+jj,m,n);
                    block[bIdx] = A[sIdx];
                }

            // DEBUG!!!
            //std::cout << "Local block size: " << mLocalBlock << "," << nLocalBlock << std::endl;
            //std::stringstream name;
            //name << "A(" << i << "," << j << "):" << std::ends;
            //print_rowmajor(block, 4, 4, name.str());
            //std::cout << std::endl;
        }
    }
    //std::cout << "-------------------------------" << std::endl << std::endl;

    // Perform source matrix allocation
    //std::cout << "------- TRANSPOSE MATRIX ---------" << std::endl;
    for(int i = 0; i < nBlocks; i++)   // i = row
    {
        for(int j = 0; j < mBlocks; j++) // j = column
        {
            // Allocate memory for the block
            float *block = (float *) calloc (16, sizeof(float));
            transpose_blocks[ROW_MAJOR(i,j,nBlocks,mBlocks)] = block;

            // Compute the dimensions of this block
            int nLocalBlock = MIN(n - (4 * i), 4);
            int mLocalBlock = MIN(m - (4 * j), 4);

            // Copy and transpose, ii = row, jj = column
            for(int ii = 0; ii < nLocalBlock; ii++)
                for(int jj = 0; jj < mLocalBlock; jj++)
                {
                    int bIdx = COLUMN_MAJOR(ii,jj,4,4);
                    int sIdx = ROW_MAJOR((i*4)+ii,(j*4)+jj,n,m);
                    block[bIdx] = A[sIdx];
                }

            // DEBUG!!!
            //std::cout << "Local block size: " << mLocalBlock << "," << nLocalBlock << std::endl;
            //std::stringstream name;
            //name << "A(" << i << "," << j << "):" << std::ends;
            //print_columnmajor(block, 4, 4, name.str());
            //std::cout << std::endl;
        }
    }
    //std::cout << "-------------------------------" << std::endl << std::endl;

    // Perform multiplication (i = row, j = column, k = element)
    for(int i = 0; i < mBlocks; i++)
    {
        for(int j = 0; j < mBlocks; j++)
        {
            for(int k = 0; k < nBlocks; k++)
            {
                // Compute the addresses
                int sIdx = ROW_MAJOR(i,k,mBlocks,nBlocks);
                int tIdx = ROW_MAJOR(k,j,nBlocks,mBlocks);
                int dIdx = COLUMN_MAJOR(i,j,mBlocks,mBlocks);

                // Perform multiply and accumulate
                block_multiply_accumulate(source_blocks[sIdx], transpose_blocks[tIdx], destination_blocks[dIdx]);
            }
        }
    }

    // Display results
    //std::cout << "------- RESULT MATRIX ---------" << std::endl;

    // Glue everything together, i = row, j = column
    for(int j = 0; j < mBlocks; j++)
    {
        for(int i = 0; i < mBlocks; i++)
        {
            // Get the block
            float *block = destination_blocks[COLUMN_MAJOR(i,j,mBlocks,mBlocks)];

            // Compute the dimensions of this block
            int mLocalBlock = MIN(m - (4 * i), 4);
            int nLocalBlock = MIN(m - (4 * j), 4);

            // Iterate through columns
            for(int jj = 0; jj < nLocalBlock; jj++)
            {
                memcpy((void *) (C + COLUMN_MAJOR((i*4),(j*4)+jj,m,m)), (void *) (block + COLUMN_MAJOR(0,jj,4,4)), sizeof(float)*mLocalBlock);
            }

            // Display the block
            //std::stringstream name;
            //name << "C(" << i << "," << j << "):" << std::ends;
            //print_columnmajor(block, 4, 4, name.str());
            //std::cout << std::endl;

            // Free the block
            free(block);
        }
    }
    //std::cout << "-------------------------------" << std::endl << std::endl;

    // Free memory
    for(int i = 0; i < mBlocks * nBlocks; i++)
    {
        free(source_blocks[i]);
        free(transpose_blocks[i]);
    }
    free(source_blocks);
    free(transpose_blocks);
    free(destination_blocks);
}

/* method provided by the code */
void sgemm_reference( int m, int n, float *A, float *C)
{
    #pragma omp parallel for
    for( int i = 0; i < m; i++ )
        for( int k = 0; k < n; k++ )
            for( int j = 0; j < m; j++ )
                C[i+j*m] += A[i+k*m] * A[j+k*m];
}

// Test
int main ()
{
  size_t m = 5;
  size_t n = 5;
  /*float  matrixA[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};*/
  float  matrixA[] = {1.0f, 2.0f, 3.0f, 4.0f, 24.f,
                      5.0f, 6.0f, 7.0f, 8.0f, 69.f,
                      9.0f, 10.0f, 11.0f, 12.0f, 69.f,
                      13.0f, 14.0f, 15.0f, 16.0f, 69.f,
                      42.f, 69.f, 69.f, 69.f, 69.f};
  /*float  matrixA[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f, 16.0f,
                      69.f, 69.f, 69.f, 69.f, 69.f};*/
  float *A  = (float *) matrixA;
  float  matrixC[m*m];
  float *C  = (float *) matrixC;

  // Print the source matrix
  std::cout << "Computing C = A * transpose(A)" << std::endl;
  print_columnmajor(A, m, n, "A []:");

  // Compute and print the transpose matrix
  //Transpose(A, At, m, n, m, n);
  //print_columnmajor(At, n, m, "A'[]:");

  // Perform multiplication with the original method
  memset((void *) C, 0, sizeof(float) * m * m);
  sgemm_reference(m, n, A, C);
  print_columnmajor(C, m, m, "C []:");
  std::cout << std::endl;

  // Zero the destination array
  memset((void *) C, 0, sizeof(float) * m * m);

  // Iterate through the columns of the matrix
  sgemm(m,n,A,C);
  //block_multiply_accumulate(At, At, C);

  // Output resulting arrays
  std::cout << "Result using SSE optimized algorithm" << std::endl;
  print_columnmajor(C, m, m, "C []:");
  std::cout << std::endl;

  return 0;
}
