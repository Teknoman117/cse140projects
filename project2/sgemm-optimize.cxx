#include <iostream>
#include <cstdlib>
#include <cstring>

// In order to effectively use

// Transpose a matrix
inline void Transpose(float * A, float * B, int m, int n, int bmPadded)
{
    // Convert from column major to row major
    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
        {
            B[(i*bmPadded) + j] = A[i + (j*n)];
        }
    }
}

// Man why the fuck is this matrix in column major order god damn it.
// since we want to multiply A by transpose(A), the operands would all
// be in sequential memory.  Fuck.

// Multiply A matrix by its transpose
extern "C" void sgemm( int m, int n, int d, float *A, float *C )
{
    // Recompute boundaries of the matrix (align to 4, for sse)
    int mPadded = (m & ~0x03) + ((m & 0x03) ? 4 : 0);
    int nPadded = (n & ~0x03) + ((n & 0x03) ? 4 : 0);

    // Allocate and transpose the source matrix (to get it column major)
    float * nA = (float *) __builtin_assume_aligned(aligned_alloc(16, sizeof(float) * mPadded * nPadded), 16);
    memset((void *) nA, 0, sizeof(float) * mPadded * nPadded);
    Transpose(A, nA, m, n, mPadded);

    // "j" will be the column used in rmatrix
    #pragma omp parallel for
    for(size_t j = 0; j < n; j++)
    {
        // "i" will be the row used in lmatrix
        for(size_t i = 0; i < n; i++)
        {
            // This loop tends to favor the columns of the destination matrix
            // and the rows of the transpose matrix in cache

            // Because we are multiplying by A by transpose(A), on a row
            // major matrix, the columns in the transpose are the sequential memory
            // rows of the first

            // Possibly write an inline assembly piece of code that
            // does 4 destination results together.  Manly due to the
            // sequential nature of adding

            // Compute the value in the destination matrix
            float t = 0.0f;
            for(size_t k = 0; k < m; k += 4)
            {
                t += nA[(i*mPadded) + k + 0] * nA[(j*mPadded) + k + 0];
                t += nA[(i*mPadded) + k + 1] * nA[(j*mPadded) + k + 1];
                t += nA[(i*mPadded) + k + 2] * nA[(j*mPadded) + k + 2];
                t += nA[(i*mPadded) + k + 3] * nA[(j*mPadded) + k + 3];
            }

            // store in destination matrix
            C[i + (j * n)] = t;
        }
    }

    // Destroy working memory
    //free(nA);
}
