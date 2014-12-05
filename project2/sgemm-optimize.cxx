#include <iostream>
#include <cstdlib>
#include <cstring>

#include <xmmintrin.h>

/*//for threading
#define THREADS 8

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

using std::mutex;
using std::thread;
using std::condition_variable;
using std::unique_lock;
using std::promise;
using std::future;

class semaphore
{
private:
    mutex mutex_;
    condition_variable condition_;
    long count_;

public:
    semaphore() : count_(0)   {}

    void increase()
    {
      unique_lock<mutex> lock(mutex_);
      ++count_;
      condition_.notify_one();
    }

    void decrease()
    {
      unique_lock<mutex> lock(mutex_);
      while(count_ <= 0)
        condition_.wait(lock);
        --count_;
    }
};

// simple task queue
class workqueue
{
private:
    std::queue<std::function<void ()> > _work;
    mutex               _mutex;
    condition_variable  _condition;

public:
    workqueue()
    {
        // Fire up some threads for this work queue
        for(size_t i = 0; i < THREADS; i++)
        {
            thread t([this] ()
            {
                // Work object
                std::function<void ()> task;

                // Do FOREVER
                while(1)
                {
                    // Get some work to do
                    {
                        unique_lock<mutex> lock(_mutex);
                        while(_work.empty())
                            _condition.wait(lock);
                        task = _work.front();
                        _work.pop();
                    }
                    task();
                }
            });
            t.detach();
        }
    }

    // Insert work
    void add(std::function<void ()> task)
    {
        unique_lock<mutex> lock(_mutex);
        _work.push(task);
        _condition.notify_one();
    }
};

// Pool of available threads to schedule work on
static workqueue threadpool;*/

// Transpose a matrix (respecting the padding)
inline void Transpose(float * A, float * B, int m, int n, int bmPadded, int bnPadded)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
            B[(i*bnPadded) + j] = A[i + (j*m)];
}

// Computes C = A * transpose(A).  m and n must be multiples of 4. A is row major, C is column major
void atimestransposea( int m, int n, float *A, float *C )
{
    // Iterate through the columns of the matrix
    //#pragma omp parallel for
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

// Computes C = A * transpose(A).  m and n must be multiples of 4
/*inline void atimestransposea_threaded( int m, int n, float *A, float *C )
{
    // Create a mutex to lock the critical sections of the model loader worker threads
    semaphore * deadThreads = new semaphore();

    // Compute how many threads to use
    size_t threadCount = THREADS;
    if(threadCount > m) threadCount = m;

    // Compute boundaries of stuff
    size_t workPerThread = m / threadCount;
    for(size_t a = 0; a < threadCount; a++)
    {
        threadpool.add([a,deadThreads,workPerThread,m,n,A,C] ()
        {
            // Iterate through the columns of the matrix
            size_t workStart = workPerThread * a;
            size_t workEnd = workStart + workPerThread;
            for(size_t i = workStart; i < workEnd; i++)
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

            // We are complete
            deadThreads->increase();
        });
    }

    // Wait for all threads to finish
    for(int a = 0; a < THREADS; a++)
    {
        deadThreads->decrease();
    }
}*/

// Multiply A matrix by its transpose
extern "C" void sgemm( int m, int n, float *A, float *C )
{
    // Recompute boundaries of the matrix (align to 4, for sse)
    int mPadded = (m & ~0x03) + ((m & 0x03) ? 4 : 0);
    int nPadded = (n & ~0x03) + ((n & 0x03) ? 4 : 0);

    // Convert from column major to row major and pad for non-divisible-by-4 matrices
    float *At = (float *) malloc (sizeof(float) * mPadded * nPadded);
    memset((void *) At, 0, sizeof(float) * mPadded * nPadded);
    Transpose(A, At, m, n, mPadded, nPadded);

    // The matrix does not have favorable dimensions
    if((m % 4) || (n % 4))
    {
        // Allocate padded destination matrix
        float *Cpadded = (float *) malloc (sizeof(float) * mPadded * mPadded);

        // Perform multiplication
        atimestransposea(mPadded, nPadded, At, Cpadded);

        // Perform a copy of Cpadded into C matrix (optimized for column major matrices,
        // hopefully we'll get SSE accelerated memcpy)
        for(int j = 0; j < m; j++)
            memcpy((void *) (C + (m*j)), (void *) (Cpadded + (mPadded*j)), sizeof(float) * m);

        // Cleanup
        free(Cpadded);
    }

    // Otherwise, this is an optimal case where padding is not required
    else
    {
        atimestransposea(m, n, At, C);
    }

    // Release the transpose matrix
    free(At);
}
