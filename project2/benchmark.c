#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <emmintrin.h>

/* Your function must have the following signature: */

void sgemm( int m, int n, int d, float *A, float *C );


/* The reference code */
void sgemm_reference( int m, int n, float *A, float *C )
{
  /*#pragma omp parallel for
  for( int i = 0; i < n; i++ )
    for( int k = 0; k < m; k++ )
      for( int j = 0; j < n; j++ )
	C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];*/
  #pragma omp parallel for
  for( int i = 0; i < n; i++)
    for( int k = 0; k < m; k++)
      for( int j = 0; j < n; j++)
        C[j+(i*n)] += A[j+(k*n)] * A[i+(k*n)];
}

/* The benchmarking program */

int main( int argc, char **argv )
{
  srand(time(NULL));

  	for( int n = 32; n < 100; n = n+1+n/3 )
//	for( int n = 40; n <  41; n = n+1+n/3 )
	{
  /* Try different m */
  for( int m = 32; m <  100; m = m+1+m/3 )
//for( int m = 48; m <  49; m = m+1+m/3 )
  {
    /* Allocate and fill 2 random matrices A, C */
    float *A = (float*) malloc( (n+m) * n * sizeof(float) );
    float *C = (float*) malloc( n * n * sizeof(float) );
    float *C_ref = (float*) malloc( n * n * sizeof(float) );

    for( int i = 0; i < (n+m)*n; i++ ) A[i] = 2 * drand48() - 1;

    /* measure Gflop/s rate; time a sufficiently long sequence of calls to eliminate noise */
    double Gflop_s, seconds = -1.0;
    for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2 )
    {
      /* warm-up */
      sgemm( m, n,m, A, C );

      /* measure time */
      struct timeval start, end;
      gettimeofday( &start, NULL );
      for( int i = 0; i < n_iterations; i++ )
	sgemm( m,n,m, A, C );
      gettimeofday( &end, NULL );
      seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);

      /* compute Gflop/s rate */
      Gflop_s = 2e-9 * n_iterations * m * n * n / seconds;
    }

    printf( "%d by %d matrix with strip size %d \t %g Gflop/s\n", n, n,m, Gflop_s );

    /* Ensure that error does not exceed the theoretical error bound */

    /* Set initial C to 0 and do matrix multiply of A*B */
    memset( C, 0, sizeof( float ) * n * n );
    sgemm( m,n,m, A, C );

    /* Subtract A*B from C using standard sgemm and reference (note that this should be 0 to within machine roundoff) */
    memset( C_ref, 0, sizeof( float ) * n * n );
    sgemm_reference( m,n,A,C_ref );

    /* Subtract the maximum allowed roundoff from each element of C */
    for( int i = 0; i < n*n; i++ ) C[i] -= C_ref[i] ;

    /* After this test if any element in C is still positive something went wrong in square_sgemm */
    for( int i = 0; i < n * n; i++ )
      if( C[i] > 0.0001 ) {
  	printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
  	printf( "Off by: %f, from the reference: %f, at n = %d, m = %d\n",C[i], C_ref[i], n, m );
  	return -1;
      }

    /* release memory */
    free( C_ref );
    free( C );
    free( A );
  }
	}
  return 0;
}
