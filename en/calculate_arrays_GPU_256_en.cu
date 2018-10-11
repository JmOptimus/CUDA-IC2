/*
  CUDA C program
  @author Juan Manuel Tortajada
  @mail ai.robotics.inbox@gmail.com
*/

#include <iostream>
#include <sys/time.h>

__global__
void calculate_arrays_GPU( int n, float *A, float *B, float *C, float *D, float *E, float *F, float *G, float *H, float *K ){

  int unique_thread_id = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    /*
      Ensure no thread accesses to memory areas out of each array bounds
      (special case in which the length of each array is not multiple of
      the number of threads per block)
    */
    if(unique_thread_id < n){
      C[unique_thread_id] = A[unique_thread_id] + B[unique_thread_id];
      F[unique_thread_id] = D[unique_thread_id] - E[unique_thread_id];
      G[unique_thread_id] = K[unique_thread_id] * H[unique_thread_id];
    }
}

int main(void){
  float elapsed_time_ms;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 1<<20;  //  1 048 574 elements

  float *A, *B, *C, *D, *E, *F, *G, *H, *K; //  Host's arrays
  float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_G, *d_H, *d_K; //  GPU's arrays

  //  Allocate memory at the Host

  A = (float *)malloc( N*sizeof(float) );
  B = (float *)malloc( N*sizeof(float) );
  C = (float *)malloc( N*sizeof(float) );
  D = (float *)malloc( N*sizeof(float) );
  E = (float *)malloc( N*sizeof(float) );
  F = (float *)malloc( N*sizeof(float) );
  G = (float *)malloc( N*sizeof(float) );
  H = (float *)malloc( N*sizeof(float) );
  K = (float *)malloc( N*sizeof(float) );


  //  Allocate memory at the Device (GPU)

  cudaMalloc( &d_A, N*sizeof(float) );
  cudaMalloc( &d_B, N*sizeof(float) );
  cudaMalloc( &d_C, N*sizeof(float) );
  cudaMalloc( &d_D, N*sizeof(float) );
  cudaMalloc( &d_E, N*sizeof(float) );
  cudaMalloc( &d_F, N*sizeof(float) );
  cudaMalloc( &d_G, N*sizeof(float) );
  cudaMalloc( &d_H, N*sizeof(float) );
  cudaMalloc( &d_K, N*sizeof(float) );


  //  Random array initialization([0,1e6])

  for(int i = 0; i < N; i++){
    A[i] = 1e6 * ( rand()/RAND_MAX );
    B[i] = 1e6 * ( rand()/RAND_MAX );
    C[i] = 1e6 * ( rand()/RAND_MAX );
    D[i] = 1e6 * ( rand()/RAND_MAX );
    E[i] = 1e6 * ( rand()/RAND_MAX );
    F[i] = 1e6 * ( rand()/RAND_MAX );
    G[i] = 1e6 * ( rand()/RAND_MAX );
    H[i] = 1e6 * ( rand()/RAND_MAX );
    K[i] = 1e6 * ( rand()/RAND_MAX );
  }

  //  Data copy from Host to GPU

  cudaMemcpy( d_A, A, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_B, B, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_C, C, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_D, D, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_E, E, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_F, F, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_G, G, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_H, H, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_K, K, N*sizeof(float), cudaMemcpyHostToDevice );


  cudaEventRecord(start);  //  Start GPU timer
  /*
    Number of blocks: 1
    Threads per block: 256
  */
  calculate_arrays_GPU<<<1, 256>>>( N, d_A, d_B, d_C, d_D, d_E, d_F, d_G, d_H, d_K );
  cudaEventRecord(stop);  //  Stop GPU timer
  cudaEventSynchronize(stop);  // Wait for the GPU to finish with the data


  //  Once data is ready, the result is copied from GPU to Host

  cudaMemcpy( A, d_A, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( B, d_B, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( D, d_D, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( E, d_E, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( F, d_F, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( G, d_G, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( H, d_H, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( K, d_K, N*sizeof(float), cudaMemcpyDeviceToHost );

  cudaEventElapsedTime( &elapsed_time_ms, start, stop ); //  Calculate elapsed time [ms]
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //  Test 10 first elements
  for(int i = 0; i < 10; i++){
    bool test1 = ( C[i] == A[i] + B[i] );
    bool test2 = ( F[i] == D[i] - E[i] );
    bool test3 = ( G[i] == K[i] * H[i] );

    printf( "\nC[%i] = A[%i] + B[%i] :%s\n", i, i, i, test1 ? "correct" : "failed");
    printf( "F[%i] = D[%i] - E[%i] :%s\n", i, i, i, test2 ? "correct" : "failed");
    printf( "G[%i] = K[%i] * H[%i] :%s\n", i, i, i, test3 ? "correct" : "failed");

  }

  printf("\nElapsed time (GPU : kernel calculate_arrays_GPU) : %f ms\n\n", elapsed_time_ms);

  // Free memory (GPU)
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
  cudaFree(d_E);
  cudaFree(d_F);
  cudaFree(d_G);
  cudaFree(d_H);
  cudaFree(d_K);

    // Free memory (CPU)
  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(F);
  free(G);
  free(H);
  free(K);

  return 0;
}
