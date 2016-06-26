
/*

  MBLBP gentleboosted Decision Stump WeakLearner

  Usage
  ------

  [model , wnew] = mblbp_gentle_weaklearner(X , y , w);

  
  Inputs
  -------

  X                                     Features matrix (d x N) in UINT8 format (see mblbp function)
  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
  w                                     Current Weigths (1 x N) at stage m in DOUBLE format

  Outputs
  -------

  Model                                 Model output (4 x 1) for current stage n of the classifier's premodel
              featureIdx                Feature indexe of the best weaklearner
			  th                        Optimal Threshold parameters )
			  a                         WeakLearner's weight in R 
			  b                         Offset 



  wnew                                  Updated weights (1 x N) at stage m+1 



  To compile
  ----------


  mex  -output mblbp_gentle_weaklearner.dll mblbp_gentle_weaklearner.c

  mex  -g -DOMP COMPFLAGS="$COMPFLAGS /openmp"  -output mblbp_gentle_weaklearner.dll mblbp_gentle_weaklearner.c -I"C:\Program Files\Microsoft Visual Studio 8\VC\lib"

  mex  -f mexopts_intel10.bat -output mblbp_gentle_weaklearner.dll mblbp_gentle_weaklearner.c


  mex  -v -DOMP -f mexopts_intel10.bat -output mblbp_gentle_weaklearner.dll mblbp_gentle_weaklearner.c



  Example 1
  ---------

  clear
  load viola_24x24

  [Ny , Nx , N]      = size(X);
  options.F          = mblbp_featlist(Ny , Nx);
  H                  = mblbp(X , options);
  options.indexF     = int32(0:size(H , 1)-1);

  y                  = int8(y);
  w                  = (1/length(y))*ones(1 , length(y));

  tic,[model , wnew] = mblbp_gentle_weaklearner(H , y , w , options );,toc


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Reference ""


*/

#include <math.h>
#include <mex.h>

#ifdef OMP 
 #include <omp.h>
#endif



#define huge 1e300

struct opts
{
    int           *indexF;
};


/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

void qsindex( unsigned char * , int * , int , int  );
void transpose(unsigned char *, unsigned char * , int , int);
void  gentleboost_decision_stump(unsigned char * , char * , double * , int  , int , struct opts , double * , double *);


/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )

{
    unsigned char *X ;
	char *y;
	double *wold;
	double *model ,*wnew;

	int i , d , N; 

	struct opts options;   
	mxArray *mxtemp;

	
    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) ==2) && (!mxIsEmpty(prhs[0])) && (mxIsUint8(prhs[0])) )
	{		
		X           = (unsigned char *)mxGetData(prhs[0]);	
		d           = mxGetM(prhs[0]);
		N           = mxGetN(prhs[0]);
	}
	else
	{
		mexErrMsgTxt("X must be (d x N) in UINT8 format");
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
		mxtemp                            = mxGetField(prhs[3] , 0 , "indexF");	
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != d)
			{
				mexErrMsgTxt("index F(1 x d), in int16 format");   

			}
			options.indexF                = (int *) mxGetData(mxtemp);
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
	else
	{
		options.indexF                = (int *) mxMalloc(d*sizeof(int));
		for (i = 0 ; i < d ; i++)
		{
			options.indexF[i]         = i;
		}
	}   

		
    /*------------------------ Outputs ----------------------------*/
			
	plhs[0]              = mxCreateNumericMatrix(4 , 1 , mxDOUBLE_CLASS,mxREAL);
	model                = mxGetPr(plhs[0]);
	
	
	plhs[1]              = mxCreateNumericMatrix(1 , N , mxDOUBLE_CLASS,mxREAL);
	wnew                 = mxGetPr(plhs[1]);

   /*------------------------ Main Call ----------------------------*/

	gentleboost_decision_stump(X , y , wold , d , N , options , model, wnew );


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

void  gentleboost_decision_stump(unsigned char *X , char *y , double *wold , int d , int N , struct opts options , double *model , double *wnew)								 
{

	int *indexF = options.indexF;

	int i , j ;	
	int indN , Nd = N*d , ind , N1 = N - 1 , featuresIdx_opt;
	int indice ;
	
	double *wtemp ;
	unsigned char *Xt, *xtemp;
	char *ytemp;

	int *idX;
	
	double atemp , btemp  , Eyw , fm  , sumwyy , error , errormin, th_opt , a_opt , b_opt;
	double Syw, Sw , temp;
	
    idX          = (int *)malloc(Nd*sizeof( int));
	Xt           = (unsigned char *)malloc(Nd*sizeof(unsigned char ));

#ifdef OMP 
#pragma omp parallel private(xtemp,wtemp,ytemp)
#endif
	xtemp        = (unsigned char *)malloc(N*sizeof(unsigned char ));
	wtemp        = (double *)malloc(N*sizeof(double));
	ytemp        = (char *)malloc(N*sizeof(char));
	

#ifdef OMP 
#pragma omp parallel for firstprivate(i) lastprivate(j,indN) shared(idX)
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
	
	transpose(X , Xt , d , N);


	/* Sorting data to speed up computation */


#ifdef OMP 
/* #pragma omp parallel for private(j , indN) shared(Xt , idX , d , N , N1) */
#pragma omp parallel for
#endif

	for(j = 0 ; j < d ; j++)
	{
		indN        = j*N;
		qsindex(Xt + indN , idX + indN , 0 , N1);		
	}
	
	Eyw              = 0.0;
	sumwyy           = 0.0;
	
	for(i = 0 ; i < N ; i++)
	{
		temp        = y[i]*wold[i];
		Eyw        += temp;
		sumwyy     += y[i]*temp;
	}
	
	errormin         = huge;
	indN             = 0;
	

#ifdef OMP 
/* #pragma omp parallel for firstprivate(i,ind,indice,xtemp,ytemp,wtemp,btemp,atemp,error) lastprivate(j,indN) shared(d,Eyw,sumwyy,errormin,indexF,idX,Xt,wold,y,N,N1,a_opt,b_opt,th_opt,featuresIdx_opt) reduction (+:Sw,Syw) */
#pragma omp parallel for  private(j,i,ind,indice,atemp,error,errormin) shared(d,N,N1,indexF,idX,Xt,wold,y,Eyw,sumwyy,featuresIdx_opt,th_opt,a_opt,b_opt) reduction (+:Sw,Syw) 
/* #pragma omp parallel for default(none) private(error,i,ind,indice,atemp,btemp,wtemp,xtemp,ytemp,Sw,Syw) lastprivate(j,indN) shared(d,Eyw,sumwyy,errormin,indexF,idX,Xt,wold,y,N,N1,a_opt,b_opt,th_opt,featuresIdx_opt) */
#endif

	for(j = 0 ; j < d  ; j++)
	{
		indN          = j*N;	
		if (indexF[j] != -1)
		{
/*
#ifdef OMP 
#pragma omp parallel for private(i,ind,indice) shared (Xt,y,wold,xtemp,ytemp,wtemp,idX,indN)
#endif
*/
			for(i = 0 ; i < N ; i++)
			{
				ind         = i + indN;	
				indice      = idX[ind];
				xtemp[i]    = Xt[ind];
				wtemp[i]    = wold[indice];
				ytemp[i]    = y[indice];
			}

			Sw              = 0.0;
			Syw             = 0.0;
/*
#ifdef OMP 
#pragma omp parallel for private(i,btemp,atemp,error) shared (j,xtemp,wtemp,ytemp,sumwyy,Eyw,errormin,featuresIdx_opt,a_opt,b_opt,th_opt,N,N1) reduction(+:Sw,Syw) 
#endif
*/
			for(i = 0 ; i < N ; i++)
			{
				Sw        += wtemp[i];	
				Syw       += ytemp[i]*wtemp[i];
				btemp      = Syw/Sw;

				if(Sw != 1.0)
				{	
					atemp  = (Eyw - Syw)/(1.0 - Sw) - btemp;	
				}
				else
				{
					atemp  = (Eyw - Syw) - btemp;
				}
				error   = sumwyy - 2.0*atemp*(Eyw - Syw) - 2.0*btemp*Eyw + (atemp*atemp + 2.0*atemp*btemp)*(1.0 - Sw) + btemp*btemp;
/*
#ifdef OMP 
                #pragma omp critical
                {
#endif
*/
				if(error < errormin)					
				{
					errormin        = error;	
					featuresIdx_opt = j;
					if(i < N1)
					{			
						th_opt     = (xtemp[i] + xtemp[i + 1])/2;	
					}
					else
					{
						th_opt     = xtemp[i];	
					}
					a_opt          = atemp;
					b_opt          = btemp;		
				}
/*
#ifdef OMP 
				}
#endif
*/
			}
		}
	}
	
	ind              = featuresIdx_opt*N;

#ifdef OMP 
#pragma omp parallel for private(i,fm) shared (Xt,y,wold,a_opt,b_opt,th_opt,ind,N)
#endif

	for (i = 0 ; i < N ; i++)
	{	
		fm           = a_opt*(Xt[i + ind] > th_opt) + b_opt;	
		wnew[i]      = wold[i]*exp(-y[i]*fm);
	}

	model[0]         = (double) (featuresIdx_opt + 1);
	model[1]         = th_opt;
	model[2]         = a_opt;
	model[3]         = b_opt;

				
	free(idX);
	free(Xt);
	free(xtemp);
	free(wtemp);
	free(ytemp);	
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */

void qsindex (unsigned char  *a, int *index , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi , ind;
    unsigned char x=a[(lo+hi)/2] , h;

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



#ifdef OMP 
		if (lo<j) qsindex(a , index , lo , j);
		if (i<hi) qsindex(a , index , i , hi);

#else
		if (lo<j) qsindex(a , index , lo , j);
		if (i<hi) qsindex(a , index , i , hi);
#endif
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void transpose(unsigned char *A, unsigned char *B , int m , int n)
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

