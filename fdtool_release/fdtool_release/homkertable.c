
/*

Homogeneous Feature Kernel Map table

Usage
------

PSI = homkertable([options]);


Inputs
-------

options 
        n                      Appoximation order (n>0, default n = 1)
        L                      Sampling step (default L = 0.5);
        kerneltype             0 for intersection kernel, 1 for Jensen-shannon kernel, 2 for Chi2 kernel (default kerneltype = 0)
        minexponent            Minimum exponent value (default minexponent = -20)
        maxexponent            Maximum exponent value (default minexponent = 8)
		numsubdiv              Number of subdivisions (default numsubdiv = 8);

Outputs
-------

table                          table (1 x (2*n+1)*(maxexponent - minexponent + 1)*numsubdiv) in double format

To compile
----------

mex  -g -output homkertable.dll homkertable.c
mex  -f mexopts_intel10.bat -output homkertable.dll homkertable.c

Example 1
---------

options.n            = 1;
options.L            = 1;
options.kerneltype   = 0;
options.minexponent  = -20;
options.maxexponent  = 8;
options.numsubdiv    = 8;

options.homtable     = homkertable(options );

plot(1:(2*options.n+1)*(options.maxexponent - options.minexponent + 1)*options.numsubdiv , options.homtable)
title(sprintf('L = %4.3f, n = %d' , options.L , options.n))



Author : Sébastien PARIS : sebastien.paris@lsis.org
-------  Date : 09/23/2010

References [1] A. Vedaldi and A. Zisserman, "Efficient Additive Kernels via Explicit Feature Maps", 
                in Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2010
           [2] http://www.vlfeat.org/

*/

#include <math.h>
#include <mex.h>

#define PI 3.14159265358979323846
#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif

struct opts
{
	int            n;
	double         L;
	int            kerneltype;
	int            numsubdiv;
	int            minexponent;
	int            maxexponent;
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void homkertable(struct opts  , double * );

/*-------------------------------------------------------------------------------------------------------------- */
#ifdef MATLAB_MEX_FILE
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{  
	double *table;
	struct opts options = {1 , 0.5, 0 , 8 , -20 , 8};
	mxArray *mxtemp;
	double *tmp , temp;
	int tempint;

	/* Input 2  */

	if ((nrhs > 0) && !mxIsEmpty(prhs[0]) )
	{
		mxtemp                            = mxGetField(prhs[0] , 0 , "n");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 1) )
			{
				mexPrintf("n > 0, force to 1\n");	
				options.n                 = 1;
			}
			else
			{
				options.n                 = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[0] , 0 , "L");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			temp                          = tmp[0];

			if( (temp < 0.0) )
			{
				mexPrintf("L >= 0, force to 0.5\n");	
				options.L                 = 0.5;
			}
			else
			{
				options.L                 = temp;
			}
		}

		mxtemp                            = mxGetField(prhs[0] , 0 , "kerneltype");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0) ||  (tempint > 2))
			{
				mexPrintf("kerneltype = {0,1,2}, force to 0\n");	
				options.kerneltype        = 1;
			}
			else
			{
				options.kerneltype        = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[0] , 0 , "numsubdiv");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 1) )
			{
				mexPrintf("numsubdiv > 0 , force to 8\n");	
				options.numsubdiv         = 8;
			}
			else
			{
				options.numsubdiv         = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[0] , 0 , "minexponent");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			options.minexponent           = (int) tmp[0];
		}

		mxtemp                            = mxGetField(prhs[0] , 0 , "maxexponent");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < options.minexponent) )
			{
				mexPrintf("maxexponent > minexponent , force to 8\n");	
				options.maxexponent       = options.minexponent + 2;
			}
			else
			{
				options.maxexponent       = tempint;
			}
		}
	}

	/*----------------------- Outputs -------------------------------*/

	plhs[0]                               =  mxCreateDoubleMatrix(1 , (2*options.n+1)*(options.maxexponent - options.minexponent + 1)*options.numsubdiv , mxREAL);
	table                                 =  mxGetPr(plhs[0]);

	/*------------------------ Main Call ----------------------------*/
	
	homkertable(options , table );
	
	/*--------------------------- Free memory -----------------------*/
}

#else


#endif

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void homkertable(struct opts options , double *table )
{
	int n = options.n, kerneltype = options.kerneltype, numsubdiv = options.numsubdiv, minexponent = options.minexponent, maxexponent = options.maxexponent;
	double L = options.L , subdiv = 1.0 / numsubdiv;
	int exponent;
	unsigned int i,j,co=0;
	double x, logx, Lx, sqrtLx, Llogx, lambda ;
	double kappa, kappa0 , sqrtkappa0, sqrt2kappa ;
	double mantissa ;

	/* table initialization */

	if (kerneltype == 0)
	{
		kappa0          = 2.0/PI;
		sqrtkappa0      = sqrt(kappa0) ;
	}
	else if (kerneltype == 1)
	{
		kappa0          = 2.0/log(4.0);
		sqrtkappa0      = sqrt(kappa0) ;
	}
	else if (kerneltype == 2)
	{
		sqrtkappa0      = 1.0 ;
	}

	for (exponent  = minexponent ; exponent <= maxexponent ; ++exponent) 
	{
		mantissa        = 1.0;
		for (i = 0 ; i < numsubdiv ; ++i , mantissa += subdiv) 
		{
			x           = ldexp(mantissa, exponent);
			Lx          = L * x ;
			logx        = log(x);
			sqrtLx      = sqrt(Lx);
			Llogx       = L*logx;
			table[co++] = (sqrtkappa0 * sqrtLx);

			for (j = 1 ; j <= n ; ++j) 
			{
				lambda = j * L;
				if (kerneltype == 0)
				{
					kappa   = kappa0 / (1.0 + 4.0*lambda*lambda) ;
				}
				else if (kerneltype == 1)
				{
					kappa   = kappa0 * 2.0 / (exp(PI * lambda) + exp(-PI * lambda)) / (1.0 + 4.0*lambda*lambda) ;
				}
				else if (kerneltype == 2)
				{
					kappa   = 2.0 / (exp(PI * lambda) + exp(-PI * lambda)) ;
				}
				sqrt2kappa  = sqrt(2.0 * kappa)* sqrtLx ;
				table[co++] = (sqrt2kappa * cos(j * Llogx)) ;
				table[co++] = (sqrt2kappa * sin(j * Llogx)) ;
			}
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
