
#include<flame.h>
#include<stdio.h>

int main( int argc, char *argv[] )
{
  float **data = NULL;
  int i, j, N = 0, M = 0;
  FILE *fin;
  Flame *flame;
  if( argc <2 ){
    printf( "No input file\n" );
    return 0;
  }
  fin = fopen( argv[1], "r" );
  fscanf( fin, "%i %i\n", & N, & M );

  printf( "Reading dataset with %i rows and %i columns\n", N, M );

  data = malloc( N * sizeof(float*) );
  for( i=0; i<N; i++ ){
    data[i] = malloc( M * sizeof(float) );
    for( j=0; j<M; j++ ) fscanf( fin, "%f ", & data[i][j] );
  }
  flame = Flame_New();
  Flame_SetDataMatrix( flame, data, N, M, 0 );

  printf( "Detecting Cluster Supporting Objects ..." );
  fflush( stdout );
  Flame_DefineSupports( flame, 10, -2.0 );
  printf( "done, found %i\n", flame->cso_count );

  printf( "Propagating fuzzy memberships ... " );
  fflush( stdout );
	Flame_LocalApproximation( flame, 500, 1e-6 );
  printf( "done\n" );

  printf( "Defining clusters from fuzzy memberships ... " );
  fflush( stdout );
	Flame_MakeClusters( flame, -1.0 );
  printf( "done\n" );

  for( i=0; i<=flame->cso_count; i++){
    if( i == flame->cso_count )
      printf( "\nCluster outliers, with %6i members:\n", flame->clusters[i].size );
    else
      printf( "\nCluster %3i, with %6i members:\n", i+1, flame->clusters[i].size );
    for( j=0; j<flame->clusters[i].size; j++){
      if( j ){
        printf( "," );
        if( j % 10 ==0 ) printf( "\n" );
      }
      printf( "%5i", flame->clusters[i].array[j] );
    }
    printf( "\n" );
  }
  return 0;
}
