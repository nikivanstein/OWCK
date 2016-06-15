/*===================================================================
  The standard implementation of FLAME data clustering algorithm.

  FLAME (Fuzzy clustering by Local Approximation of MEmberships)
  was first described in:
  "FLAME, a novel fuzzy clustering method for the analysis of DNA
  microarray data", BMC Bioinformatics, 2007, 8:3.
  Available from: http://www.biomedcentral.com/1471-2105/8/3
  
  Copyright(C) 2007, Fu Limin (phoolimin@gmail.com).
  All rights reserved.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. The origin of this software must not be misrepresented; you must 
     not claim that you wrote the original software. If you use this 
     software in a product, an acknowledgment in the product 
     documentation would be appreciated but is not required.
  3. Altered source versions must be plainly marked as such, and must
     not be misrepresented as being the original software.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
===================================================================*/

#ifndef __FLAME_H
#define __FLAME_H

/* Since data for clustering are usually noisy,
 * so it is not very necessary to have EPSILON extremely small.
 */
#define EPSILON 1E-9

typedef struct IndexFloat IndexFloat;
typedef struct Flame Flame;
typedef struct IntArray IntArray;

struct IntArray
{
	int *array;
	int  size;
	int  bufsize;
};

/* For sorting and storing the orignal indices. */
struct IndexFloat
{
	int   index;
	float value;
};

/* Sort until the smallest "part" items are sorted. */
void PartialQuickSort( IndexFloat *data, int first, int last, int part );

float Flame_Euclidean( float *x, float *y, int m );
float Flame_Cosine( float *x, float *y, int m );
float Flame_Pearson( float *x, float *y, int m );
float Flame_UCPearson( float *x, float *y, int m );
float Flame_SQPearson( float *x, float *y, int m );
float Flame_DotProduct( float *x, float *y, int m );
float Flame_Covariance( float *x, float *y, int m );
float Flame_Manhattan( float *x, float *y, int m );
float Flame_CosineDist( float *x, float *y, int m );
float Flame_PearsonDist( float *x, float *y, int m );
float Flame_UCPearsonDist( float *x, float *y, int m );
float Flame_SQPearsonDist( float *x, float *y, int m );
float Flame_DotProductDist( float *x, float *y, int m );
float Flame_CovarianceDist( float *x, float *y, int m );

enum DistSimTypes
{
	DST_USER = 0,
	DST_EUCLID ,
	DST_COSINE ,
	DST_PEARSON ,
	DST_UC_PEARSON ,
	DST_SQ_PEARSON ,
	DST_DOT_PROD ,
	DST_COVARIANCE ,
	DST_MANHATTAN ,
	DST_NULL
};
typedef float (*DistFunction)( float *x, float *y, int m );

extern const DistFunction basicDistFunctions[];

enum FlameObjectTypes
{
	OBT_NORMAL ,
	OBT_SUPPORT ,
	OBT_OUTLIER
};

struct Flame
{
	int simtype;

	/* Number of objects */
	int N;

	/* Number of K-Nearest Neighbors */
	int K;

	/* Upper bound for K defined as: sqrt(N)+10 */
	int KMAX;

	/* Stores the KMAX nearest neighbors instead of K nearest neighbors
	 * for each objects, so that when K is changed, weights and CSOs can be
	 * re-computed without referring to the original data.
	 */
	int   **graph;
	/* Distances to the KMAX nearest neighbors. */
	float **dists;

	/* Nearest neighbor count.
	 * it can be different from K if an object has nearest neighbors with
	 * equal distance. */
	int    *nncounts;
	float **weights;

	/* Number of identified Cluster Supporting Objects */
	int cso_count;
	char *obtypes;

	float **fuzzyships;
	
	/* Number of clusters including the outlier group */
	int count;
	/* The last one is the outlier group. */
	IntArray *clusters;
	
	DistFunction distfunc;
};



/* Create a structure for FLAME clustering, and set all fields to zero. */
Flame* Flame_New();

void Print_Clusters(Flame *self,float* fuzzy,int size);

/* Free allocated memory, and set all fields to zero. */
void Flame_Clear( Flame *self );

/* Set a NxM data matrix, and compute distances of type T.
 * 
 * If T==DST_USER or T>=DST_NULL, and Flame::distfunc member is set,
 * then Flame::distfunc is used to compute the distances;
 * Otherwise, Flame_Euclidean() is used. */
void Flame_SetDataMatrix( Flame *self, float *data, int N, int M, int T );

/* Set a pre-computed NxN distance matrix. */
void Flame_SetDistMatrix( Flame *self, float *data[], int N );

/* Define knn-nearest neighbors for each object 
 * and the Cluster Supporting Objects (CSO). 
 * 
 * The actual number of nearest neighbors could be large than knn,
 * if an object has neighbors of the same distances.
 *
 * Based on the distances of the neighbors, a density can be computed
 * for each object. Objects with local maximum density are defined as
 * CSOs. The initial outliers are defined as objects with local minimum
 * density which is less than mean( density ) + thd * stdev( density );
 */
void Flame_DefineSupports( Flame *self, int knn, float thd );

/* Local Approximation of fuzzy memberships.
 * Stopped after the maximum steps of iterations;
 * Or stopped when the overall membership difference between
 * two iterations become less than epsilon. */
void Flame_LocalApproximation( Flame *self, int steps, float epsilon );

/* Construct clusters.
 * If 0<thd<1:
 *   each object is assigned to all clusters in which
 *   it has membership higher than thd; if it can not be assigned
 *   to any clusters, it is then assigned to the outlier group.
 * Else:
 *   each object is assigned to the group (clusters/outlier group)
 *   in which it has the highest membership. */
void Flame_MakeClusters( Flame *self, float thd );


#endif
