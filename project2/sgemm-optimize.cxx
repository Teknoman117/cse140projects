#include <iostream>
#include <cstdlib>

// In order to effectively use 

// Transpose a matrix
inline void Transpose(float * A, float * B, int m, int n)
{
    // Convert from column major to row major
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            _B[(i*n) + j] = _A[i + (j*m)];
        }
    }
}

// Man why the fuck is this matrix in column major order god damn it.
// since we want to multiply A by transpose(A), the operands would all
// be in sequential memory.  Fuck.

// Multiply A matrix by its transpose
extern "C" void sgemm( int m, int n, int d, float *A, float *C )
{
    // Recompute boundaries of the matrix


    // Allocate a matrix to hold the transposed one in aligned memory


    // Generate a row major order of the matrix (cause its moar l337)
    float *nA = Transpose(A, n, m);

    // "j" will be the column used in rmatrix
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

            // Compute the value in the destination matrix
            float t = 0.0f;
            for(size_t k = 0; k < m; k++)
                t += nA[(i*m) + k] * nA[(j*m) + k];

            // store in destination matrix
            C[i + (j * n)] = t;
        }
    }

    // Destroy working memory
    delete nA;
}
