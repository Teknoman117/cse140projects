void sgemm( int m, int n, int d, float *A, float *C )
{
    // this addresses columns of the destination matrix
    /*for( int i = 0; i < n; i++ )
    {
        // this addresses col
        for( int k = 0; k < m; k++ )
        {
            // this accesses rows of the destination matrix
            for( int j = 0; j < n; j++ )
            {
                C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
            }
        }
    }*/
    #pragma omp parallel for
    for( int i = 0; i < n; i++)
      for( int k = 0; k < m; k++)
        for( int j = 0; j < n; j++)
          C[j+(i*n)] += A[j+(k*n)] * A[i+(k*n)];
}
