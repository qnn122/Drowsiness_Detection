
/*

  Haar Adaboosted Decision Stump WeakLearner

  Usage
  ------

  [model , wnew]  = haar_ada_weaklearner(II , y , w , [options]);

  
  Inputs
  -------

  II                                    Images Integral (Ny x Nx x N) standardized in DOUBLE format

  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format

  w                                     Current Weigths (1 x N) at stage m in DOUBLE format

  options

         rect_param                     Features rectangles parameters (10 x nR), where nR is the total number of rectangles for the patterns.
                                        (default Vertical(2 x 1) [1 ; -1] and Horizontal(1 x 2) [-1 , 1] patterns) 
										rect_param(: , i) = [if ; wp ; hp ; nrif ; nr ; xr ; yr ; wr ; hr ; sr], where
										if     index of the current Haar's feature. ip = [1,...,nF], where nF is the total number of Haar's features
										wp     width of the current rectangle pattern defining current Haar's feature
										hp     height of the current rectangle pattern defining current Haar's feature
										nrif   total number of rectangles for the current Haar's feature if
										nr     index of the current rectangle in the current Haar's feature, nr=[1,...,nrif]
										xr,yr  top-left coordinates of the current rectangle of the current Haar's feature
										wr,hr  width and height of the current rectangle of the current Haar's feature
										sr     weights of the current rectangle of the current Haar's feature 

         F                              Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle

	     indexF                         Index of accesible weaklearners (default index = int32(0:nF-1));

If compiled with the "OMP" compilation flag

	     num_threads                    Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)



  Outputs
  -------


  
  Model                                 Model output (4 x 1) for current stage n of the classifier's premodel
              featureIdx                Feature indexe of the best weaklearner
			  th                        Optimal Threshold parameters )
			  a                         WeakLearner's weight  in R (at = ct*pt, where pt = polarity)
			  b                         Zeros 

  wnew                                  Updated weights (1 x N) at stage m+1 


  To compile
  ----------


  mex  -output haar_ada_weaklearner.dll haar_ada_weaklearner.c

  mex  -f mexopts_intel10.bat -output haar_ada_weaklearner.dll haar_ada_weaklearner.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output haar_ada_weaklearner.dll haar_ada_weaklearner.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib" 


  Example 1
  ---------

  clear, close all
  load viola_24x24
  options            = load('haar_dico_2.mat');
  [Ny , Nx , N]      = size(X);
  options.F          = haar_featlist(Ny , Nx , options.rect_param);
  options.indexF     = int32(0:size(options.F , 2)-1);

  II                 = image_integral_standard(X);
  y                  = int8(y);
  w                  = (1/length(y))*ones(1 , length(y));

 
  tic,[model , wnew] = haar_ada_weaklearner(II , y , w , options);,toc




 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999

              [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
			      ICB07

			  [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
			  [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004


 Changelog :  - Add OpenMP support
 ----------   - Add indexF vector for force to not select twice the same weaklearner


*/

#include <math.h>
#include "mex.h"

#define huge 1e300
#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))

#ifdef OMP 
 #include <omp.h>
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif


struct opts
{
	double        *rect_param;
	int            nR;
	unsigned int  *F;
	int            nF;
    int           *indexF;
#ifdef OMP 
    int            num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
double Area(double * , int , int , int , int , int );
double haar_feat(double *  , int  , double * , unsigned int * , int , int , int );
void qsindex( double * , int * , int , int  );
void  adaboost_decision_stump(double *, char *, double *, int , int , int , struct opts , double *, double *);

/*---------------------------------------------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    double *II , *wold;
	char *y;
	const int *dimsII;
	double *model , *wnew;	
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	int i , Ny , Nx , N; 	
	struct opts options;   
	mxArray *mxtemp;
    int tempint;
	double *tmp;

	options.nR           = 4;

#ifdef OMP 
    options.num_threads  = -1;
#endif

    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) ==3) && (!mxIsEmpty(prhs[0])) && (mxIsDouble(prhs[0])) )
	{
		II          = mxGetPr(prhs[0]);
		dimsII      = mxGetDimensions(prhs[0]);	
		Ny          = dimsII[0];
		Nx          = dimsII[1];
		N           = dimsII[2];
	}
	else
	{
		mexErrMsgTxt("Integral Image II must be (Ny x Nx x N) in DOUBLE format");		
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
		mxtemp                             = mxGetField( prhs[3] , 0, "rect_param" );
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 10)
			{
				mexErrMsgTxt("rect_param must be (10 x nR)");
			}

			options.rect_param            = mxGetPr(mxtemp);              
			options.nR                    = mxGetN(mxtemp);;
		}
		else
		{
			options.rect_param            = (double *)mxMalloc(40*sizeof(double));	
			for (i = 0 ; i < 40 ; i++)
			{		
				options.rect_param[i]     = rect_param_default[i];
			}			
		}

		mxtemp                             = mxGetField( prhs[3] , 0, "F" );
		if(mxtemp != NULL)
		{

			options.F                     = (unsigned int *) mxGetData(mxtemp);
			options.nF                    = mxGetN(mxtemp);
		}
		else		
		{
			options.nF                    = number_haar_features(Ny , Nx , options.rect_param , options.nR);
			options.F                     = (unsigned int *)mxMalloc(5*options.nF*sizeof(unsigned int));
			haar_featlist(Ny , Nx , options.rect_param , options.nR , options.F);	
		}

		mxtemp                            = mxGetField(prhs[3] , 0 , "indexF");	
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != options.nF)
			{
				mexErrMsgTxt("index F(1 x nF), in int16 format");   

			}
			options.indexF                = (int *) mxGetData(mxtemp);
		}
		else
		{
			options.indexF                = (int *) mxMalloc(options.nF*sizeof(int));
			for (i = 0 ; i < options.nF ; i++)
			{
				options.indexF[i]         = i;
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
		options.rect_param            = (double *)mxMalloc(40*sizeof(double));	
		for (i = 0 ; i < 40 ; i++)
		{		
			options.rect_param[i]     = rect_param_default[i];
		}		

		options.nF                    = number_haar_features(Ny , Nx , options.rect_param , options.nR);
		options.F                     = (unsigned int *)mxMalloc(5*options.nF*sizeof(unsigned int));
		haar_featlist(Ny , Nx , options.rect_param , options.nR , options.F);	 

		options.indexF                = (int *) mxMalloc(options.nF*sizeof(int));
		for (i = 0 ; i < options.nF ; i++)
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
				
		adaboost_decision_stump(II , y , wold , Ny , Nx , N , options , model, wnew );
		
	/*----------------- Free Memory --------------------------------*/

	if ( (nrhs > 3) && !mxIsEmpty(prhs[3]) )
	{
		if ( (mxGetField( prhs[3] , 0 , "rect_param" )) == NULL )
		{
			mxFree(options.rect_param);
		}
		if ( (mxGetField( prhs[3] , 0 , "F" )) == NULL )
		{
			mxFree(options.F);
		}
		if ( mxGetField( prhs[3] , 0 , "indexF" ) == NULL )	
		{
			mxFree(options.indexF);
		}
	}
	else
	{
		mxFree(options.rect_param);
		mxFree(options.F);
		mxFree(options.indexF);
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  adaboost_decision_stump(double *II , char *y , double *wold , int Ny , int Nx , int N , struct opts options , double *model , double *wnew)							  
{
	double *rect_param = options.rect_param;
	unsigned int *F = options.F;
	int *indexF = options.indexF;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	int nF = options.nF , nR = options.nR;
	int i , j;
	int NyNx = Ny*Nx, ind , N1 = N - 1 , featuresIdx_opt;
	double  Tplus , Tminus , Splus , Sminus , Errormin , errm  , cm , Errplus , Errminus , wtemp , a_opt ;
	char ytemp;
	double *xtemp , z , th_opt ;
	char  *h;
	int *index;

	h                   = (char *)malloc(N*sizeof(char));

#ifdef OMP 
    num_threads          = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif

	Tplus               = 0.0;
	Tminus              = 0.0;

#ifdef OMP 
#pragma omp parallel for private(i) shared (wold,y,N) reduction (+:Tplus,Tminus) 
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

	Errormin            = huge;

#ifdef OMP 
#pragma omp parallel default(none) private(xtemp,index,ytemp,wtemp,j,i,ind,Errplus,Errminus) shared(N,N1,NyNx,Ny,nR,nF,indexF,II,wold,y,rect_param,F,featuresIdx_opt,th_opt,a_opt,Errormin,Tplus,Tminus) reduction (+:Splus,Sminus) 
#endif
	{
		xtemp               = (double *)malloc(N*sizeof(double ));
		index               = (int *)malloc(N*sizeof(int));

#ifdef OMP 
#pragma omp for nowait
#endif
		for(j = 0 ; j < nF  ; j++)
		{	
			if (indexF[j] != -1)
			{	
				for(i = 0 ; i < N ; i++)				
				{
					index[i]     = i;	
					xtemp[i]     = haar_feat(II + i*NyNx , j , rect_param , F , Ny , nR , nF);	
				}
				qsindex(xtemp , index , 0 , N1);

				Splus            = 0.0;
				Sminus           = 0.0;

				for(i = 0 ; i < N ; i++)
				{
					ind         = index[i];
					ytemp       = y[ind];
					wtemp       = wold[ind];
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
					if(ytemp == 1)
					{
						Splus  += wtemp;
					}
					else
					{
						Sminus += wtemp;
					}	
				}
			}
		}
		free(index);
		free(xtemp);
	}

	errm             = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(i,z) shared (II,wold,y,h,a_opt,th_opt,featuresIdx_opt,rect_param,F,N,NyNx,Ny,nR,nF) reduction (+:errm) 
#endif
	for (i = 0 ; i < N ; i++)
	{
		z            = haar_feat(II + i*NyNx , featuresIdx_opt , rect_param , F , Ny , nR , nF);
		h[i]         = a_opt*sign(z - th_opt);
		if(y[i] != h[i])
		{	
			errm    += wold[i];	
		}
	}

	cm              = 0.5*log((1.0 - errm)/errm);

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (wnew,wold,y,h,N,cm)
#endif
	for (i = 0 ; i < N ; i++)
	{
		wnew[i]     = wold[i]*exp(-y[i]*h[i]*cm);
	}

	model[0]        = (double) (featuresIdx_opt + 1);
	model[1]        = th_opt;
	model[2]        = a_opt*cm;
	model[3]        = 0.0;

	free(h);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double haar_feat(double *II , int featidx , double *rect_param , unsigned int *F , int Ny , int nR , int nF)
{
	int x , xr , y , yr , w , wr , h , hr , r , s  ,  R , indR , indF = featidx*6;
	int coeffw , coeffh;
	double val = 0.0;
	
	x     = F[1 + indF];
	y     = F[2 + indF];
	w     = F[3 + indF];
	h     = F[4 + indF];
	indR  = F[5 + indF];
	R     = (int) rect_param[3 + indR];
		
	for (r = 0 ; r < R ; r++)
	{		
		coeffw  = w/(int)rect_param[1 + indR];
		coeffh  = h/(int)rect_param[2 + indR];
		xr      = x + coeffw*(int)rect_param[5 + indR];
		yr      = y + coeffh*(int)rect_param[6 + indR];
		wr      = coeffw*(int)(rect_param[7 + indR]);
		hr      = coeffh*(int)(rect_param[8 + indR]);
		s       = rect_param[9 + indR];
		val    += s*Area(II , xr  , yr  , wr , hr , Ny);
		indR   += 10;
	}
	return val;		
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void qsindex (double  *a, int *index , int lo, int hi)
{
	/*  lo is the lower index, hi is the upper index
	of the region of array a that is to be sorted
	*/
	int i=lo, j=hi , ind;
	double x=a[(lo+hi)/2] , h;
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
void haar_featlist(int ny , int nx , double *rect_param , int nR , unsigned int *F )
{
	int  r , indF = 0 , indrect = 0 , currentfeat = 0 , temp , W , H , w , h , x , y;
	int nx1 = nx + 1, ny1 = ny + 1;
	
	for (r = 0 ; r < nR ; r++)
	{
		temp            = (int) rect_param[0 + indrect];
		if(currentfeat != temp)
		{
			currentfeat = temp;		
			W           = (int) rect_param[1 + indrect];
			H           = (int) rect_param[2 + indrect];
				
			for(w = W ; w < nx1 ; w +=W)
			{
				for(h = H ; h < ny1 ; h +=H)	
				{
					for(y = 0 ; y + h < ny1 ; y++)
					{
						for(x = 0 ; x + w < nx1 ; x++)
						{
							F[0 + indF]   = currentfeat;							
							F[1 + indF]   = x;
							F[2 + indF]   = y;
							F[3 + indF]   = w;
							F[4 + indF]   = h;
							F[5 + indF]   = indrect;
							indF         += 6;	
						}
					}
				}
			}
		}
		indrect        += 10;		
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int number_haar_features(int ny , int nx , double *rect_param , int nR)
{
	int i , temp , indrect = 0 , currentfeat = 0 , nF = 0 , h , w;
	int Y , X;
	int nx1 = nx + 1, ny1 = ny + 1;
	
	for (i = 0 ; i < nR ; i++)
	{
		temp            = (int) rect_param[0 + indrect];
		if(currentfeat != temp)
		{
			currentfeat = temp;			
			w           = (int) rect_param[1 + indrect];			
			h           = (int) rect_param[2 + indrect];
			X           = (int) floor(nx/w);
			Y           = (int) floor(ny/h);
			nF         += (int) (X*Y*(nx1 - w*(X+1)*0.5)*(ny1 - h*(Y+1)*0.5));
		}
		
		indrect   += 10;
	}
	return nF;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double Area(double *II , int x , int y , int w , int h , int Ny)
{
	int h1 = h-1 , w1 = w-1 , x1 = x-1, y1 = y-1;
	
	if( (x == 0) && (y==0))
	{
		return (II[h1 + w1*Ny]);	
	}
	if( (x==0) ) 
	{
		return(II[(y+h1) + w1*Ny] - II[y1 + w1*Ny]);
	}
	if( (y==0) ) 
	{	
		return(II[h1 + (x+w1)*Ny] - II[h1 + x1*Ny]);	
	}
	else
	{	
		return (II[(y+h1) + (x+w1)*Ny] - (II[y1 + (x+w1)*Ny] + II[(y+h1) + x1*Ny]) + II[y1 + x1*Ny]);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
