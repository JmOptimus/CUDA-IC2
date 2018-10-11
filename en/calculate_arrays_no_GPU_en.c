/*
  @author Juan Manuel Tortajada
  @mail ai.robotics.inbox@gmail.com
*/

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

void calculate_arrays_no_GPU( int n, float *A, float *B, float *C, float *D, float *E, float *F, float *G, float *H, float *K ){
  for(int i = 0; i < n; i++){
    C[i] = A[i] + B[i];
    F[i] = D[i] - E[i];
    G[i] = K[i] * H[i];
  }
}

int main(void){
  struct timespec inicio;
  struct timespec fin;
  int N = 1<<20;  //  1 048 574 elementos

  float *A, *B, *C, *D, *E, *F, *G, *H, *K;

  //  Reserva de memoria

  A = (float *)malloc( N*sizeof(float) );
  B = (float *)malloc( N*sizeof(float) );
  C = (float *)malloc( N*sizeof(float) );
  D = (float *)malloc( N*sizeof(float) );
  E = (float *)malloc( N*sizeof(float) );
  F = (float *)malloc( N*sizeof(float) );
  G = (float *)malloc( N*sizeof(float) );
  H = (float *)malloc( N*sizeof(float) );
  K = (float *)malloc( N*sizeof(float) );

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

  clock_gettime( CLOCK_REALTIME, &inicio ); //  Inicio del temporizador

  calculate_arrays_no_GPU( N, A, B, C, D, E, F, G, H, K );

  clock_gettime(CLOCK_REALTIME, &fin);  //  Parada del temporizador
  double tiempo_transcurrido_s = (fin.tv_sec - inicio.tv_sec)+ (fin.tv_nsec - inicio.tv_nsec)/1e9;  //  Calculo del tiempo transcurrido [s]

  //  Comprueba los primeros 10 elementos de los tres vectores resultado
  for(int i = 0; i < 10; i++){
    bool test1 = ( C[i] == A[i] + B[i] );
    bool test2 = ( F[i] == D[i] - E[i] );
    bool test3 = ( G[i] == K[i] * H[i] );

    printf( "\nC[%i] = A[%i] + B[%i] :%s\n", i, i, i, test1 ? "correcto" : "erroneo");
    printf( "F[%i] = D[%i] - E[%i] :%s\n", i, i, i, test2 ? "correcto" : "erroneo");
    printf( "G[%i] = K[%i] * H[%i] :%s\n", i, i, i, test3 ? "correcto" : "erroneo");

  }


  printf("\nTiempo transcurrido (ejecucion OperarVectores) : %f ms\n", tiempo_transcurrido_s*1e3);

  // Liberacion de memoria
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
