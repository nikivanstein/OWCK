/* example.i */
 %module flame
 %{
#define SWIG_FILE_WITH_INIT
 /* Put header files here or function declarations like below */
#include"stdio.h"
#include"stdlib.h"
#include"string.h"
#include"math.h"

#include"flame.h"
 %}
 
%include "numpy.i"
%init %{
import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {
    (float *data, int N, int M)
};

%apply (float* ARGOUT_ARRAY1, int DIM1) {
    (float *fuzzy, int size)
};
%include "flame.h"