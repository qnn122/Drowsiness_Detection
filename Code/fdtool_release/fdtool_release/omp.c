
// mex -v -DOMP -f mexopts_intel10.bat -output omp.dll omp.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"

#include "mex.h"
#ifdef OMP
#include <omp.h>
#else
#warning "OpenMP not enabled. Use -fopenmp (>gcc-4.2) or
-openmp (icc) for speed enhancements on SMP machines."
#endif

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const
				 mxArray* prhs[])
{
	int j;

	/* Some basic stuff */
	mexPrintf("Max threads %d.\n", omp_get_num_procs());
	omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel
	{
		mexPrintf("Hello from thread num %d.\n",omp_get_thread_num());
	}

	/* OMP loop */
#pragma omp for schedule(dynamic,10) nowait
	for (j=0; j<100; j++)
		mexPrintf("%d ", j);

	mexPrintf("\n");

}
