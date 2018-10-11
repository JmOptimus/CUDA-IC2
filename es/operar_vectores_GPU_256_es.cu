/*
  Programa C CUDA
  @author Juan Manuel Tortajada
  @mail ai.robotics.inbox@gmail.com
*/

#include <iostream>
#include <sys/time.h>

__global__
void operar_vectores_GPU( int n, float *A, float *B, float *C, float *D, float *E, float *F, float *G, float *H, float *K ){

  int indice_hilo_unico = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    /*
      Asegura que para una longitud de los vectores
      no multiplo del nº de hilos por bloque
      no existan hilos accediendo a posiciones de memoria fuera del vector
      (debordamiento)
    */
    if(indice_hilo_unico < n){
      C[indice_hilo_unico] = A[indice_hilo_unico] + B[indice_hilo_unico];
      F[indice_hilo_unico] = D[indice_hilo_unico] - E[indice_hilo_unico];
      G[indice_hilo_unico] = K[indice_hilo_unico] * H[indice_hilo_unico];
    }
}

int main(void){
  float tiempo_transcurrido_ms;
  cudaEvent_t inicio,fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);

  int N = 1<<20;  //  1 048 574 elementos

  float *A, *B, *C, *D, *E, *F, *G, *H, *K; //  Vectores en el Host
  float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_G, *d_H, *d_K; //  Vectores en el dispositivo(GPU)

  //  Reserva de memoria en el Host

  A = (float *)malloc( N*sizeof(float) );
  B = (float *)malloc( N*sizeof(float) );
  C = (float *)malloc( N*sizeof(float) );
  D = (float *)malloc( N*sizeof(float) );
  E = (float *)malloc( N*sizeof(float) );
  F = (float *)malloc( N*sizeof(float) );
  G = (float *)malloc( N*sizeof(float) );
  H = (float *)malloc( N*sizeof(float) );
  K = (float *)malloc( N*sizeof(float) );


  //  Reserva de memoria en el dispositivo (GPU)

  cudaMalloc( &d_A, N*sizeof(float) );
  cudaMalloc( &d_B, N*sizeof(float) );
  cudaMalloc( &d_C, N*sizeof(float) );
  cudaMalloc( &d_D, N*sizeof(float) );
  cudaMalloc( &d_E, N*sizeof(float) );
  cudaMalloc( &d_F, N*sizeof(float) );
  cudaMalloc( &d_G, N*sizeof(float) );
  cudaMalloc( &d_H, N*sizeof(float) );
  cudaMalloc( &d_K, N*sizeof(float) );


  //  Inicializacion de vectores (valores aleatorios [0,1e6])

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

  //  Copia de datos del Host al Dispositivo(GPU)

  cudaMemcpy( d_A, A, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_B, B, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_C, C, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_D, D, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_E, E, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_F, F, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_G, G, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_H, H, N*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_K, K, N*sizeof(float), cudaMemcpyHostToDevice );


  cudaEventRecord(inicio);  //  Inicio del temporizador en la GPU
  /*
    Numero de bloques: 1
    Hilos por bloque: 256
  */
  operar_vectores_GPU<<<1, 256>>>( N, d_A, d_B, d_C, d_D, d_E, d_F, d_G, d_H, d_K );
  cudaEventRecord(fin);  //  Parada del temporizador en la GPU
  cudaEventSynchronize(fin);  // Espera a que los datos esten listos


  //  Una vez los datos estan listos, se copia el resultado del dispositivo(GPU) al Host

  cudaMemcpy( A, d_A, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( B, d_B, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( D, d_D, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( E, d_E, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( F, d_F, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( G, d_G, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( H, d_H, N*sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( K, d_K, N*sizeof(float), cudaMemcpyDeviceToHost );

  cudaEventElapsedTime( &tiempo_transcurrido_ms, inicio, fin ); //  Calculo del tiempo transcurrido [ms]
  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);

  //  Comprueba los primeros 10 elementos de los tres vectores resultado
  for(int i = 0; i < 10; i++){
    bool test1 = ( C[i] == A[i] + B[i] );
    bool test2 = ( F[i] == D[i] - E[i] );
    bool test3 = ( G[i] == K[i] * H[i] );

    printf( "\nC[%i] = A[%i] + B[%i] :%s\n", i, i, i, test1 ? "correcto" : "erroneo");
    printf( "F[%i] = D[%i] - E[%i] :%s\n", i, i, i, test2 ? "correcto" : "erroneo");
    printf( "G[%i] = K[%i] * H[%i] :%s\n", i, i, i, test3 ? "correcto" : "erroneo");

  }

  printf("\nTiempo transcurrido (GPU : kernel operarVectores) : %f ms\n\n", tiempo_transcurrido_ms);

  // Liberacion de memoria (GPU)
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
  cudaFree(d_E);
  cudaFree(d_F);
  cudaFree(d_G);
  cudaFree(d_H);
  cudaFree(d_K);

  // Liberacion de memoria (CPU)
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
