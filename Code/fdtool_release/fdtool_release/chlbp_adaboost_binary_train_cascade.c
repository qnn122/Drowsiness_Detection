
/*

  Train Circular Histogram Local Binary Pattern with adaboost classifier for binary problem

  Usage
  ------

  param = chlbp_adaboost_binary_train_cascade(X , y , [options]);


  
  Inputs
  -------

  X                                     Features matrix (d x N) in UINT32 format

  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
  options
         T                              Number of weak learners (default T = 100)
	     weaklearner                    Choice of the weak learner used in the training phase (default weaklearner = 2)
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
         premodel                       Classifier's premodels parameter up to n-1 stage (4 x Npremodels)(default premodel = [] for stage n=1)

If compiled with the "OMP" compilation flag
	     num_threads                    Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  param                                 param output (4 x T) for current stage n of the classifier's premodel
              featureIdx                Feature indexes of the T best weaklearners (1 x T)
			  th                        Optimal Threshold parameters (1 x T)
			  a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)
			  b                         Zeros (1 x T), i.e. b = zeros(1 , T)


  To compile
  ----------


  mex  -output chlbp_adaboost_binary_train_cascade.dll chlbp_adaboost_binary_train_cascade.c

  mex  -f mexopts_intel10.bat -output chlbp_adaboost_binary_train_cascade.dll chlbp_adaboost_binary_train_cascade.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output chlbp_adaboost_binary_train_cascade.dll chlbp_adaboost_binary_train_cascade.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


  Example 1
  ---------

  clear, close all
  load viola_24x24

  Ny                                 = 24;
  Nx                                 = 24;
  options.T                          = 100;
  options.N                          = [8 , 4];
  options.R                          = [1 , 1];
  options.map                        = zeros(2^max(options.N) , length(options.N));
  mapping                            = getmapping(options.N(1),'u2');
  options.map(1:2^options.N(1) , 1)  = mapping.table';
  options.map(1:2^options.N(2) , 2)  = (0:2^options.N(2)-1)';
  options.shiftbox                   = cat(3 , [Ny , Nx ; 1 , 1] , [16 , 16 ; 4 , 4]);


  H                                  = chlbp(X , options);
  figure(1)
  imagesc(H)

  index                              = randperm(length(y));

  y                                  = int8(y);
  tic,options.param                  = chlbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);,toc
  [yest , fx]                        = chlbp_adaboost_binary_predict_cascade(H , options);
  indp                               = find(y == 1);
  indn                               = find(y ==-1);

  tp                                 = sum(yest(indp) == y(indp))/length(indp)
  fp                                 = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf                               = sum(yest == y)/length(y)


  [tpp , fpp]                        = basicroc(y , fx);

  figure(2)
  plot(fpp , tpp , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])


  [dum , ind]                        = sort(y , 'descend');
  figure(3)
  plot(fx(ind))

  figure(4)
  plot(abs(options.param(3 , :)) , 'linewidth' , 2)
  grid on
  xlabel('Weaklearner m')
  ylabel('|a_m|')



 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009


 Changelog :  - Add OpenMP support
 ----------   

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
#define verytiny 1e-15
#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))

struct opts
{
  int     weaklearner;
  int     T;
  double *premodel;
  int     Npremodel;

#ifdef OMP 
    int   num_threads;
#endif

};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void qsindex( unsigned int * , int * , int , int  );
void transpose(unsigned int *, unsigned int * , int , int);
void adaboost_decision_stump(unsigned int * , char * , struct opts , int , int , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{		
    unsigned int *X ;	
	char *y;
	double *param;

	int d , N ; 

	mxArray *mxtemp;
	struct opts options;
	double *tmp;
	int tempint;
	
	options.weaklearner = 2;
	options.T           = 100;
	options.Npremodel   = 0;

#ifdef OMP 
    options.num_threads = -1;
#endif
	
    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) ==2) && (!mxIsEmpty(prhs[0])) && (mxIsUint32(prhs[0])) )
	{		
		X           = (unsigned int *)mxGetData(prhs[0]);		
		d           = mxGetM(prhs[0]);
		N           = mxGetN(prhs[0]);		
	}
	else
	{
		mexErrMsgTxt("X must be (d x N) in UINT32 format");	
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
		
	if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )	
	{
		mxtemp                            = mxGetField(prhs[2] , 0 , "weaklearner");
		if(mxtemp != NULL)
		{
			
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];		
			if((tempint < 2) || (tempint > 3))
			{
				mexPrintf("weaklearner = {2}, force to 2");		
				options.weaklearner       = 2;
			}
			else
			{		
				options.weaklearner        = tempint;	
			}	
		}

		mxtemp                            = mxGetField(prhs[2] , 0 , "T");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];

			if((tempint < 0) )
			{
				mexPrintf("T > 0, force to 100");		
				options.T                = 100;
			}
			else
			{		
				options.T                = tempint;	
			}
		}

		mxtemp                           = mxGetField(prhs[2] , 0 , "premodel");
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 4)
			{	
				mexErrMsgTxt("premodel must be (4 x Npremodel) in double format");
			}
			options.premodel                      =  mxGetPr(mxtemp);
			options.Npremodel                     =  mxGetN(mxtemp);
		}
#ifdef OMP 
		mxtemp                            = mxGetField( prhs[2] , 0, "num_threads" );
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
		/*------------------------ Main Call ----------------------------*/

	if(options.weaklearner == 2)
	{	
		plhs[0]              = mxCreateNumericMatrix(4 , options.T , mxDOUBLE_CLASS,mxREAL);	
		param                = mxGetPr(plhs[0]);	
		adaboost_decision_stump(X , y , options , d , N , param);		
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  adaboost_decision_stump(unsigned int *X , char *y , struct opts options , int d , int N , double *param )						 								 
{
	double *premodel = options.premodel;
	int i , j , t , T = options.T , Npremodel = options.Npremodel;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	int indN , Nd = N*d , ind , N1 = N - 1 , featuresIdx_opt;
	int indM , indice ;
	double *w , *wtemp;
	int *Xt, *xtemp;
	char *ytemp , *h;
	int *indexF , *idX;
	double  sumw , fm  , a_opt;
	double cteN =1.0/(double)N;
	double Tplus , Tminus , Splus , Sminus , Errormin , cm , Errplus , Errminus , errm;
	int th_opt ;
	
    idX              = (int *)malloc(Nd*sizeof(int));
	Xt               = (int *)malloc(Nd*sizeof(int ));
	w                = (double *)malloc(N*sizeof(double));
	ytemp            = (char *)malloc(N*sizeof(char ));
	h                = (char *)malloc(N*sizeof(char));
	indexF           = (int *)malloc(d*sizeof(int));

#ifdef OMP 

#else
	wtemp            = (double *)malloc(N*sizeof(double));
	xtemp            = (int *)malloc(N*sizeof(int ));
	ytemp            = (char *)malloc(N*sizeof(char ));
#endif


#ifdef OMP 
    num_threads      = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif
	
	/* Transpose data to speed up computation */
	
	transpose(X , Xt , d , N);
		
	for(i = 0 ; i < N ; i++)
	{	
		w[i]             = cteN;	
	}

	for(i = 0 ; i < d ; i++)
	{	
		indexF[i]        = i;	
	}


	/* Previous premodel */
	
	indM                 = 0;
	
#ifdef OMP 
#pragma omp parallel for default(none) private(j,i,ind,featuresIdx_opt,th_opt,a_opt,fm) shared(premodel,Npremodel,N,Xt,y,w) reduction (+:indM,sumw) 
#endif
	for(j = 0 ; j < Npremodel ; j++)
	{	
		featuresIdx_opt  = ((int) premodel[0 + indM]) - 1;	
		th_opt           = (int) premodel[1 + indM];
		a_opt            = premodel[2 + indM];
		ind              = featuresIdx_opt*N;
		sumw             = 0.0;
		
		for (i = 0 ; i < N ; i++)
		{
			fm           = a_opt*sign(Xt[i + ind] - th_opt);		
			w[i]        *= exp(-y[i]*fm);
			sumw        += w[i];
		}
				
		sumw            = 1.0/(sumw + verytiny);
		for (i = 0 ; i < N ; i++)
		{
			w[i]         *= sumw;
		}	
		indM            += 4;
	}
	
	/* Sorting data to speed up computation */

#ifdef OMP 
#pragma omp parallel for firstprivate(i) lastprivate(j,indN) shared(idX,d,N)
#endif
	for(j = 0 ; j < d ; j++)
	{
		indN        = j*N;
		for(i = 0 ; i < N ; i++)
		{	
			idX[i + indN]      =  i;	
		}
	}
#ifdef OMP 
#pragma omp parallel for default(none) private(j,indN) shared(Xt,idX,d,N,N1)
#endif
	for(j = 0 ; j < d ; j++)
	{
		indN        = j*N;
		qsindex(Xt + indN , idX + indN , 0 , N1);		
	}

	indM  = 0;
	for(t = 0 ; t < T ; t++)
	{		
		Tplus            = 0.0;	
		Tminus           = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (w,y,N) reduction (+:Tplus,Tminus) 
#endif	
		for(i = 0 ; i < N ; i++)				
		{			
			if(y[i] == 1)
			{
				Tplus    += w[i];		
			}				
			else
			{	
				Tminus   += w[i];	
			}			
		}
		
		Errormin         = huge;

#ifdef OMP 
#pragma omp parallel  default(none) private(xtemp,ind,indice,ytemp,wtemp,j,i,Errplus,Errminus,indN) shared(d,N,N1,indexF,idX,Xt,w,y,featuresIdx_opt,th_opt,a_opt,Errormin,Tplus,Tminus) reduction (+:Splus,Sminus) 
#endif
		{
#ifdef OMP 
			wtemp               = (double *)malloc(N*sizeof(double));
			xtemp               = (int *)malloc(N*sizeof(int ));
	        ytemp               = (char *)malloc(N*sizeof(char ));
#else
#endif


#ifdef OMP 
#pragma omp for nowait
#endif
			for(j = 0 ; j < d  ; j++)
			{
				indN         = j*N;

				if(indexF[j] != -1)
				{
					for(i = 0 ; i < N ; i++)	
					{
						ind         = i + indN;			
						indice      = idX[ind];
						xtemp[i]    = Xt[ind];
						ytemp[i]    = y[indice];
						wtemp[i]    = w[indice];
					}

					Splus            = 0.0;
					Sminus           = 0.0;			

					for(i = 0 ; i < N ; i++)
					{
						Errplus     = Splus  + (Tminus - Sminus);			
						Errminus    = Sminus + (Tplus - Splus);

						if(Errplus  < Errormin)
						{
							Errormin        = Errplus;
							if(i < N1)
							{	
								th_opt      = (xtemp[i] + xtemp[i + 1])/2;	
							}
							else
							{
								th_opt      = xtemp[i];	
							}
							featuresIdx_opt = j;
							a_opt           = 1.0;
						}
						if(Errminus <= Errormin)
						{	
							Errormin        = Errminus;

							if(i < N1)
							{	
								th_opt      = (xtemp[i] + xtemp[i + 1])/2;	
							}
							else
							{
								th_opt      = xtemp[i];	
							}
							featuresIdx_opt = j;
							a_opt           = -1.0;
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
#ifdef OMP
			free(xtemp);
			free(wtemp);
	        free(ytemp);
#else

#endif
		}
		
		ind              = featuresIdx_opt*N;	
		errm             = 0.0;
#ifdef OMP 
#pragma omp parallel for default(none) private(i,indice) shared (idX,Xt,N,w,y,h,a_opt,th_opt,featuresIdx_opt,ind) reduction (+:errm) 
#endif
		for (i = 0 ; i < N ; i++)
		{
			indice       = idX[i+ind];
			h[indice]    = a_opt*sign(Xt[i + ind] - th_opt);
			if(y[indice] != h[indice])
			{
				errm    += w[indice];			
			}
		}
		
		cm              = 0.5*log((1.0 - errm)/errm);
		sumw            = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (w,y,h,N,cm) reduction (+:sumw)
#endif
		for (i = 0 ; i < N ; i++)		
		{
			w[i]        *= exp(-y[i]*h[i]*cm);
			sumw        += w[i];
		}
		
		sumw            = 1.0/(sumw + verytiny);
#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (w,N,sumw)
#endif	
		for (i = 0 ; i < N ; i++)
		{	
			w[i]         *= sumw;
		}
		
		indexF[featuresIdx_opt] = -1;
		param[0 + indM]         = (double) (featuresIdx_opt + 1);
		param[1 + indM]         = (double) th_opt;
		param[2 + indM]         = a_opt*cm;
		param[3 + indM]         = 0.0;	
		indM                   += 4;	
	}

	free(idX);	
	free(Xt);
	free(w);
	free(h);	
	free(indexF);

#ifdef OMP

#else
	free(ytemp);
	free(xtemp);
	free(wtemp);
#endif
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void qsindex (unsigned int  *a, int *index , int lo, int hi)
{
    int i=lo, j=hi , ind;
    unsigned int x=a[(lo+hi)/2] , h;

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
    if (lo<j) qsindex(a , index , lo , j);
    if (i<hi) qsindex(a , index , i , hi);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void transpose(unsigned int *A, unsigned int *B , int m , int n)
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
