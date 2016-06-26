
/*

  MBLBP adaboosted Decision Stump WeakLearner

  Usage
  ------

  [model , wnew] = mblbp_ada_weaklearner(X , y , w , [options]);

  
  Inputs
  -------

  X                                     Features matrix (d x N) in UINT8 format (see mblbp function)
  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
  w                                     Current Weigths (1 x N) at stage m in DOUBLE format
  options
	     indexF                         Index of accesible weaklearners (default index = int32(0:nF-1));
		 transpose                      Suppose X' as input (in order to speed up Boosting algorithm avoiding internal transposing, default tranpose = 0)

If compiled with the "OMP" compilation flag
         num_threads                     Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  Model                                 Model output (4 x 1) for current stage n of the classifier's premodel
              featureIdx                Feature indexe of the best weaklearner
			  th                        Optimal Threshold parameters )
			  a                         WeakLearner's weight in R  (at = ct*pt, where pt = polarity)
			  b                         Zero 

  wnew                                  Updated weights (1 x N) at stage m+1 

  To compile
  ----------


  mex  -output mblbp_ada_weaklearner.dll mblbp_ada_weaklearner.c

  mex  -f mexopts_intel10.bat -output mblbp_ada_weaklearner.dll mblbp_ada_weaklearner.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output mblbp_ada_weaklearner.dll mblbp_ada_weaklearner.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


  Example 1
  ---------

  clear
  load viola_24x24

  [Ny , Nx , N]      = size(X);
  options.F          = mblbp_featlist(Ny , Nx);
  options.transpose  = 0;
  H                  = mblbp(X , options);
  if(options.transpose)
    options.indexF   = int32(0:size(H , 2)-1);
  else
   options.indexF    = int32(0:size(H , 1)-1);
  end

  y                  = int8(y);
  w                  = (1/length(y))*ones(1 , length(y));

  tic,[model , wnew] = mblbp_ada_weaklearner(H , y , w , options );,toc


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Changelog :  - Add OpenMP support
 ----------   - Add indexF vector for force to not select twice the same weaklearner
              - Add transpose option

 Reference ""

*/

#include <math.h>
#include <mex.h>

#ifdef OMP 
 #include <omp.h>
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif
#define huge 1e300
#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))

struct opts
{
    int           *indexF;
	int            transpose;
#ifdef OMP 
    int            num_threads;
#endif

};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void qsindex( unsigned char * , int * , int , int  );
void transposeX(unsigned char *, unsigned char * , int , int);
void adaboost_decision_stump(unsigned char * , char * , double * , int  , int , struct opts , double * , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{

	unsigned char *X ;
	char *y;
	double *wold;
	double *model , *wnew;
	int i , d , N;
	struct opts options;   
	mxArray *mxtemp;
    int tempint;
	double *tmp;
	options.transpose   = 0;
#ifdef OMP 
    options.num_threads  = -1;
#endif

	/* Input 1  */

	if( (mxGetNumberOfDimensions(prhs[0]) ==2) && (!mxIsEmpty(prhs[0])) && (mxIsUint8(prhs[0])) )
	{	
		X           = (unsigned char *)mxGetData(prhs[0]);	
		d           = mxGetM(prhs[0]);
		N           = mxGetN(prhs[0]);
	}
	else
	{
		mexErrMsgTxt("X must be (d x N) or (N x d) in UINT8 format");		
	}

	/* Input 2  */

	if ( (nrhs > 1) && (!mxIsEmpty(prhs[1])) && (mxIsInt8(prhs[1])) )	
	{		
		y        = (char *)mxGetData(prhs[1]);	
	}
	else
	{
		mexErrMsgTxt("y must be (1 x N) in INT8 format");
	}

	/* Input 3  */
	if ( (nrhs > 2) && (!mxIsEmpty(prhs[2])) && (mxIsDouble(prhs[2])) )	
	{		
		wold      = (double *)mxGetData(prhs[2]);	
	}
	else
	{
		mexErrMsgTxt("w must be (1 x N) in DOUBLE format");
	}

    /* Input 4  */

	if ((nrhs > 3) && !mxIsEmpty(prhs[3]) )
	{	

		mxtemp                            = mxGetField( prhs[3] , 0, "transpose" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];

			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("transpose = {0,1}, force to 0");		
				options.transpose      = 0;
			}
			else
			{
				options.transpose      = tempint;	
			}			
		}

		mxtemp                            = mxGetField(prhs[3] , 0 , "indexF");	
		if(mxtemp != NULL)
		{
			if(options.transpose)
			{
				if(mxGetN(mxtemp) != N)
				{
					mexErrMsgTxt("indexF(1 x d), in int16 format");   

				}
			}
			else
			{
				if(mxGetN(mxtemp) != d)
				{
					mexErrMsgTxt("indexF(1 x d), in int16 format");   

				}
			}
			options.indexF                = (int *) mxGetData(mxtemp);
		}
		else
		{
			if(options.transpose)
			{
				options.indexF                = (int *) mxMalloc(N*sizeof(int));
				for (i = 0 ; i < N ; i++)
				{
					options.indexF[i]         = i;
				}
			}
			else
			{
				options.indexF                = (int *) mxMalloc(d*sizeof(int));
				for (i = 0 ; i < d ; i++)
				{
					options.indexF[i]         = i;
				}
			}
		}

#ifdef OMP 
		mxtemp                            = mxGetField( prhs[3] , 0, "num_threads" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			
			if((tempint < -2))
			{								
				options.num_threads       = -1;
			}
			else
			{
				options.num_threads       = tempint;	
			}			
		}
#endif
	}
	else
	{
		if(options.transpose)
		{
			options.indexF                = (int *) mxMalloc(N*sizeof(int));
			for (i = 0 ; i < N ; i++)
			{
				options.indexF[i]         = i;
			}
		}
		else
		{
			options.indexF                = (int *) mxMalloc(d*sizeof(int));
			for (i = 0 ; i < d ; i++)
			{
				options.indexF[i]         = i;
			}
		}
	}   

	/*------------------------ Outputs ----------------------------*/

	plhs[0]              = mxCreateNumericMatrix(4 ,1 , mxDOUBLE_CLASS,mxREAL);	
	model                = mxGetPr(plhs[0]);

	if(options.transpose)
	{

		plhs[1]              = mxCreateNumericMatrix(1 , d , mxDOUBLE_CLASS,mxREAL);
		wnew                 = mxGetPr(plhs[1]);

		/*------------------------ Main Call ----------------------------*/

		adaboost_decision_stump(X , y , wold , N , d , options , model, wnew );

	}
	else
	{
		plhs[1]              = mxCreateNumericMatrix(1 , N , mxDOUBLE_CLASS,mxREAL);
		wnew                 = mxGetPr(plhs[1]);

		/*------------------------ Main Call ----------------------------*/

		adaboost_decision_stump(X , y , wold , d , N , options , model, wnew );
	}

	/*----------------- Free Memory --------------------------------*/

	if ( (nrhs > 3) && !mxIsEmpty(prhs[3]) )
	{
		if ( mxGetField( prhs[3] , 0 , "indexF" ) == NULL )	
		{
			mxFree(options.indexF);
		}
	}
	else
	{
		mxFree(options.indexF);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  adaboost_decision_stump(unsigned char *X , char *y , double *wold , int d , int N , struct opts options , double *model , double *wnew)							  
{
	int *indexF = options.indexF;
	int transpose = options.transpose;
	int i , j ;
	int indN , Nd = N*d , ind , N1 = N - 1 , featuresIdx_opt;
	int indice;
	unsigned char *Xt, *xtemp;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	unsigned char th_opt;
	int *idX;
	double a_opt;
	double Tplus , Tminus , Splus , Sminus , Errormin , cm , Errplus , Errminus;
	double *wtemp , errm;
	char *ytemp , *h;

	idX          = (int *)malloc(Nd*sizeof( int));
	Xt           = (unsigned char *)malloc(Nd*sizeof(unsigned char ));
	h            = (char *)malloc(N*sizeof(char));

#ifdef OMP 
    num_threads      = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif

#ifdef OMP 
#pragma omp parallel for default(none) firstprivate(i) lastprivate(j,indN) shared(idX,d,N)
#endif
	for(j = 0 ; j < d ; j++)
	{
		indN        = j*N;
		for(i = 0 ; i < N ; i++)
		{	
			idX[i + indN]      =  i;	
		}
	}

	/* Transpose data to speed up computation */

	if(transpose)
	{
		for(i = 0 ; i < Nd ; i++)
		{
			Xt[i]          = X[i];
		}
	}
	else
	{
		transposeX(X , Xt , d , N);
	}

	/* Sorting data to speed up computation */

#ifdef OMP 
#pragma omp parallel for default(none) private(j,indN) shared(Xt,idX,d,N,N1)
#endif
	for(j = 0 ; j < d ; j++)
	{
		indN        = j*N;
		qsindex(Xt + indN , idX + indN , 0 , N1);		
	}

	Tplus            = 0.0;
	Tminus           = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (wold,y,N) reduction (+:Tplus,Tminus) 
#endif
	for(i = 0 ; i < N ; i++)				
	{
		if(y[i] == 1)
		{
			Tplus    += wold[i];
		}				
		else
		{
			Tminus   += wold[i];
		}			
	}

	Errormin         = huge;

#ifdef OMP 
#pragma omp parallel default(none) private(xtemp,wtemp,ytemp,j,i,ind,indice,Errplus,Errminus,Splus,Sminus,Tplus,Tminus,indN) shared(d,N,N1,indexF,idX,Xt,wold,y,featuresIdx_opt,th_opt,a_opt,Errormin)
#endif
	{
		xtemp        = (unsigned char *)malloc(N*sizeof(unsigned char ));
		ytemp        = (char *)malloc(N*sizeof(char ));
		wtemp        = (double *)malloc(N*sizeof(double));

#ifdef OMP 
#pragma omp for
#endif
		for(j = 0 ; j < d  ; j++)
		{
			indN         = j*N;
			if (indexF[j] != -1)
			{
				for(i = 0 ; i < N ; i++)
				{
					ind         = i + indN;
					indice      = idX[ind];
					xtemp[i]    = Xt[ind];
					ytemp[i]    = y[indice];
					wtemp[i]    = wold[indice];
				}

				Splus            = 0.0;
				Sminus           = 0.0;			

				for(i = 0 ; i < N ; i++)
				{
					Errplus     = Splus  + (Tminus - Sminus);
					Errminus    = Sminus + (Tplus - Splus);
					if(Errplus  < Errormin)
					{
#ifdef OMP                     
#pragma omp critical
#endif
						Errormin        = Errplus;
						if(i < N1)
						{
#ifdef OMP                     
#pragma omp critical
#endif
							th_opt      = (xtemp[i] + xtemp[i + 1])/2;
						}
						else
						{
#ifdef OMP                     
#pragma omp critical
#endif
							th_opt      = xtemp[i];
						}
#ifdef OMP                     
#pragma omp critical
#endif
						{
							featuresIdx_opt = j;
							a_opt           = 1.0;
						}
					}

					if(Errminus <= Errormin)
					{
#ifdef OMP                     
#pragma omp critical
#endif
						Errormin        = Errminus;

						if(i < N1)
						{
#ifdef OMP                     
#pragma omp critical
#endif
							th_opt      = (xtemp[i] + xtemp[i + 1])/2;
						}
						else
						{
#ifdef OMP                     
#pragma omp critical
#endif
							th_opt      = xtemp[i];
						}
#ifdef OMP                     
#pragma omp critical
#endif
						{
							featuresIdx_opt = j;
							a_opt           = -1.0;
						}
					}

					if(ytemp[i] == 1)
					{
						Splus  += wtemp[i];
					}
					else
					{
						Sminus += wtemp[i];
					}	
				}
			}
		}
		free(xtemp);
		free(wtemp);
		free(ytemp);
	}

	ind              = featuresIdx_opt*N;
	errm             = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(i,indice) shared (idX,Xt,wold,y,h,ind,a_opt,th_opt,featuresIdx_opt,N) reduction (+:errm) 
#endif
	for (i = 0 ; i < N ; i++)
	{
		indice       = idX[i+ind];
		h[indice]    = a_opt*sign(Xt[i + ind] - th_opt);
		if(y[indice] != h[indice])
		{
			errm    += wold[indice];			
		}
	}

	cm         = 0.5*log((1.0 - errm)/errm);

#ifdef OMP 
#pragma omp parallel for default(none)  private(i) shared (wnew,wold,y,h,N,cm)
#endif
	for (i = 0 ; i < N ; i++)
	{
		wnew[i] = wold[i]*exp(-y[i]*h[i]*cm);
	}

	model[0]   = (double) (featuresIdx_opt + 1);
	model[1]   = (double) th_opt;
	model[2]   = a_opt*cm;
	model[3]   = 0.0;

	free(idX);
	free(Xt);
	free(h);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void qsindex (unsigned char  *a, int *index , int lo, int hi)
{
	/*  lo is the lower index, hi is the upper index
	of the region of array a that is to be sorted

	*/
	int i=lo, j=hi , ind;
	unsigned char x=a[(lo+hi)/2] , h;

	/*  partition */
	do
	{    
		while (a[i]<x) i++; 
		while (a[j]>x) j--;
		if (i<=j)
		{
			h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			ind      = index[i];
			index[i] = index[j];
			index[j] = ind;
			i++; 
			j--;
		}
	}
	while (i<=j);

	/*  recursion */
	if (lo<j) qsindex(a , index , lo , j);
	if (i<hi) qsindex(a , index , i , hi);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void transposeX(unsigned char *A, unsigned char *B , int m , int n)
{  
	int i , j , jm = 0 , in;

	for (j = 0 ; j<n ; j++)
	{
		in  = 0;
		for(i = 0 ; i<m ; i++)
		{
			B[j + in] = A[i + jm];
			in       += n;
		}
		jm           += m;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */

