void sgemm( int m, int n, int d, float *A, float *C )
{
  for( int i = 0; i < n; i++ )
    for( int k = 0; k < m; k++ ) 
      for( int j = 0; j < n; j++ ) 
	C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
}
