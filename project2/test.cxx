#include <iostream>
#include <malloc.h>
#include <cstring>

#include <xmmintrin.h>

// Index into an MxN matrix.  r = row #, c = column #
#define ROW_MAJOR(r,c,M,N)    ((r)*(N))+(c)
#define COLUMN_MAJOR(r,c,M,N) ((c)*(M))+(r)
#define MIN(a,b) (a<b)?a:b

// Swizzled row major encoding: row0:0-3,row1:0-3,row2:0-3,row3:0-3,row0:4-7
#define ROW_MAJOR_SWIZZLED_BLOCK(r,c,M,N) ( (((r)&~3)*(N)) + (((c)&~3)<<2) + (((r)%4)<<2) )
#define ROW_MAJOR_SWIZZLED(r,c,M,N)       ( ROW_MAJOR_SWIZZLED_BLOCK(r,c,M,N) + ((c)%4) )

#define COLUMN_MAJOR_SWIZZLED_BLOCK(r,c,M,N) ( (((c)&~3)*(M)) + (((r)&~3)<<2) + (((c)%4)<<2) )
#define COLUMN_MAJOR_SWIZZLED(r,c,M,N)       ( COLUMN_MAJOR_SWIZZLED_BLOCK(r,c,M,N) + ((r)%4) )

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

/* print row major matrix */
void print_rowmajor_swizzled(float *A, int m, int n, std::string name)
{
    std::cout << name;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            std::cout << "\t" << A[ROW_MAJOR_SWIZZLED(i,j,m,n)];
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

/* print column major matrix */
void print_columnmajor_swizzled(float *A, int m, int n, std::string name)
{
    std::cout << name;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            std::cout << "\t" << A[COLUMN_MAJOR_SWIZZLED(i,j,m,n)];
        std::cout << std::endl;
    }
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

// Transpose a matrix and swizzle (respecting the padding)
inline void TransposeSwizzle(float * A, float * B, int m, int n, int bmPadded, int bnPadded)
{
    // i = row, j = column
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j+=4)
        {
            // Compute the swizzled address
            //size_t offset = ((i&(~0x03))*bnPadded) + ((j&(~0x03))*4) + ((i%4)*4);
            //float *block = B + offset;
            float *block = B + ROW_MAJOR_SWIZZLED_BLOCK(i,j,bmPadded,bnPadded);
            size_t len = MIN(4,n-j);

            // Perform copies
            for(size_t k = 0; k < len; k++)
                block[k] = A[COLUMN_MAJOR(i,j+k,m,n)];
        }
}

// Computes C = A * transpose(A).  m and n must be multiples of 4. A is row major, C is column major
inline void atimestransposea_swizzled( int m, int n, float *A, float *C )
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
                //__m128 aColumn = _mm_load_ps(A + (i*n) + k);
                __m128 aColumn = _mm_load_ps(A + COLUMN_MAJOR_SWIZZLED_BLOCK(k,i,n,m));

                // Load l-matrix rows
                size_t offset = ROW_MAJOR_SWIZZLED_BLOCK(j,k,m,n);
                __m128 aRow0 = _mm_load_ps(A + offset + 0);
                __m128 aRow1 = _mm_load_ps(A + offset + 4);
                __m128 aRow2 = _mm_load_ps(A + offset + 8);
                __m128 aRow3 = _mm_load_ps(A + offset + 12);

                // Multiply each row by the column
                aRow0 = aRow0 * aColumn;
                aRow1 = aRow1 * aColumn;
                aRow2 = aRow2 * aColumn;
                aRow3 = aRow3 * aColumn;

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
extern "C" void sgemm_swizzled( int m, int n, float *A, float *C )
{
    // Recompute boundaries of the matrix (align to 4, for sse)
    int mPadded = (m & ~0x03) + ((m & 0x03) ? 4 : 0);
    int nPadded = (n & ~0x03) + ((n & 0x03) ? 4 : 0);

    // Compute the transpose of the matrix
    float *At = (float *) calloc (mPadded * nPadded, sizeof(float));
    TransposeSwizzle(A, At, m, n, mPadded, nPadded);

    // The matrix does not have favorable dimensions
    if((m % 4) || (n % 4))
    {
        // Allocate new, padded matrices
        float *Cpadded = (float *) calloc (mPadded * mPadded, sizeof(float));

        // Perform multiplication
        atimestransposea_swizzled(mPadded, nPadded, At, Cpadded);

        // Perform a copy of Cpadded into C matrix (optimized for column major matrices)
        for(int j = 0; j < m; j++)
            memcpy((void *) (C + (m*j)), (void *) (Cpadded + (mPadded*j)), sizeof(float) * m);

        // Cleanup
        free(Cpadded);
    }

    // Otherwise, this is an optimal case where padding is not required
    else
    {
        atimestransposea_swizzled(m, n, At, C);
    }

    // Release the transpose matrix
    free(At);
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
    float *A  = (float *) matrixA;
    float  matrixAt[m*n];
    float *At = (float *) matrixAt;
    float  matrixC[m*m];
    float *C  = (float *) matrixC;

    // Print the source matrix
    std::cout << "Computing C = A * transpose(A)" << std::endl;
    print_columnmajor(A, m, n, "A []:");

    // Compute and print the transpose matrix
    TransposeSwizzle(A, At, m, n, m, n);
    //print_columnmajor(At, n, m, "A'[]:");
    print_columnmajor_swizzled(At, n, m, "A'[]:");
    /*std::cout << "A []:\t";
    for(size_t i = 0; i < m * n; i++)
        std::cout << At[i] << "\t";*/
    std::cout << std::endl;

    // Perform multiplication with the original method
    memset((void *) C, 0, sizeof(float) * m * m);
    sgemm_reference(m, n, A, C);
    print_columnmajor(C, m, m, "C []:");
    std::cout << std::endl;

    // Zero the destination array
    memset((void *) C, 0, sizeof(float) * m * m);

    // Iterate through the columns of the matrix
    sgemm_swizzled(m,n,A,C);

    // Output resulting arrays
    std::cout << "Result using SSE optimized algorithm" << std::endl;
    print_columnmajor(C, m, m, "C []:");
    std::cout << std::endl;

    return 0;
}
