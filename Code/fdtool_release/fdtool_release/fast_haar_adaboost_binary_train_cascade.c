
/*

  Haar features training with a fast adaboost classifier

  Usage
  ------

  param      = fast_haar_adaboost_binary_train_cascade(II , y , [options]);

  
  Inputs
  -------

  II                                    Images Integral (Ny x Nx x N) of standartized image in DOUBLE format
  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
  options
         G                              Features sparse matrix (Ny*Nx x nF) (see Haar_matG function)
         T                              Number of weak learners (default T = 100)
         weaklearner                    Choice of the weak learner used in the training phase
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
         premodel                       Classifier's premodels parameter up to n-1 stage (4 x Npremodels)(default premodel = [] for stage n=1)
	     fine_threshold                 Fine threshold estimation with a Nelder Mead optimization algorithm (yes = 1/no = 0) (default fine_threshold = 0)

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


  mex  -output fast_haar_adaboost_binary_train_cascade.dll fast_haar_adaboost_binary_train_cascade.c "C:\Program Files\MATLAB\R2007b\bin\win32\libgoto_core2-r1.26.lib" %"C:\Program Files\MATLAB\R2007b\extern\lib\win32\microsoft\libmwblas.lib" %

  mex  -f mexopts_intel10.bat -output fast_haar_adaboost_binary_train_cascade.dll fast_haar_adaboost_binary_train_cascade.c  "C:\Program Files\MATLAB\R2009b\bin\win32\libgoto_core2-r1.26.lib"

  mex  -f mexopts_intel10.bat -output fast_haar_adaboost_binary_train_cascade.dll fast_haar_adaboost_binary_train_cascade.c  "C:\Program Files\MATLAB\R2009b\extern\lib\win32\microsoft\libmwblas.lib"

  mex  -f mexopts_intel10.bat -output fast_haar_adaboost_binary_train_cascade.dll fast_haar_adaboost_binary_train_cascade.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib" 

  If OMP directive is added, OpenMP support for multicore computation

  mex  -v -DOMP -f mexopts_intel10.bat -output fast_haar_adaboost_binary_train_cascade.dll fast_haar_adaboost_binary_train_cascade.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib" 

  If OS64 directive is added in order to compile on OS 64 (for sparse matrix)

  mex  -v  -DOS64 -DOMP -f mexopts_intel11_64.bat fast_haar_adaboost_binary_train_cascade.c "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_core.lib" "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_intel_lp64.lib" "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_intel_thread.lib" -largeArrayDims

  mex  -DOS64 -g fast_haar_adaboost_binary_train_cascade.c  -largeArrayDims "C:\Program Files\MATLAB\R2010a\extern\lib\win64\microsoft\libmwblas.lib"

  Example 1
  ---------

  clear, close all
  load viola_24x24
  options                   = load('haar_dico_2.mat');
  II                        = image_integral_standard(X);
  %clear X
  y                         = int8(y);
  options.T                 = 3;
  options.fine_threshold    = 1;

  [Ny , Nx , P]             = size(II);
  options.F                 = haar_featlist(Ny , Nx , options.rect_param);
  options.G                 = Haar_matG(Ny , Nx , options.rect_param);


  N                         = 2500;
  vect                      = [1:N , 5001:5001+N-1];
  indextrain                = vect(randperm(length(vect)));
  indextest                 = (1:length(y));
  indextest(indextrain)     = [];

  ytrain                    = y(indextrain);
  ytest                     = y(indextest);



  tic,options.param         = fast_haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);,toc
  [ytest_est , fxtest]      = haar_adaboost_binary_predict_cascade(II(: , : , indextest) , options);

  indp                      = find(ytest == 1);
  indn                      = find(ytest ==-1);

  tp                        = sum(ytest_est(indp) == ytest(indp))/length(indp)
  fp                        = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn)
  perf                      = sum(ytest_est == ytest)/length(ytest)

  [tpp , fpp , threshold]   = basicroc(ytest , fxtest);

  figure(1)
  plot(fpp , tpp , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])

  figure(2)
  plot(fxtest)

  figure(3)
  plot(abs(options.param(3 , :)) , 'linewidth' , 2)
  grid on
  xlabel('Weaklearner m')
  ylabel('|a_m|')

  bestFeat                = options.param(1,1);
  %bestFeat                = 80375;

  options.F               = options.F(: , bestFeat);

  ztest                   = haar(X(:,:,indextest) , options);
  vectztest               = (min(ztest):(max(ztest)-min(ztest))/(100-1):max(ztest));

  figure(4)
  Nn                      = histc(double(ztest(indn)) , vectztest);
  Nn                      = Nn/length(ztest);
  bar(vectztest , Nn);
  set(get(gca , 'children') , 'facecolor' , [1 0 1]);
  hold on
  Np                      = histc(double(ztest(indp)) , vectztest);
  Np                      = Np/length(ztest);
  bar(vectztest , Np);
  legend(get(gca , 'children') , 'Faces' , 'Non-faces' )
  plot(repmat(options.param(2,1),1,100) , (0:(max([Np , Nn])*1.1)/(100-1):(max([Np , Nn])*1.1)) , 'k' , 'linewidth' , 3)
  hold off

  title(sprintf('Test data, Feature = %d' , bestFeat))


  ztrain                  = haar(X(:,:,indextrain) , options);
  vectztrain              = (min(ztrain):(max(ztrain)-min(ztrain))/(100-1):max(ztrain));

  indp                    = find(ytrain == 1);
  indn                    = find(ytrain ==-1);

  figure(5)
  Nn                      = histc(double(ztrain(indn)) , vectztrain);
  Nn                      = Nn/length(ztrain);
  bar(vectztrain , Nn);
  set(get(gca , 'children') , 'facecolor' , [1 0 1]);
  hold on
  Np                      = histc(double(ztrain(indp)) , vectztrain);
  Np                      = Np/length(ztrain);
  bar(vectztrain , Np);
  legend(get(gca , 'children') , 'Faces' , 'Non-faces' )
  plot(repmat(options.param(2,1),1,100) , (0:(max([Np , Nn])*1.1)/(100-1):(max([Np , Nn])*1.1)) , 'k' , 'linewidth' , 3)
  hold off

  title(sprintf('Train data, Feature = %d' , bestFeat))


  Example 2
  ---------

  clear, close all 
  load viola_24x24

  y                         = int8(y);

  Nimage                    = 110;
  nb_feats                  = 5;

  options                   = load('haar_dico_2.mat');

  II                        = image_integral_standard(X);
  I                         = X(: , : , Nimage);
  clear X
  
  [Ny , Nx , P]             = size(II);


  options.T                 = 10;
  options.F                 = haar_featlist(Ny , Nx , options.rect_param);
  options.G                 = Haar_matG(Ny , Nx , options.rect_param);
  options.fine_threshold    = 1;


  N                         = 2000;
  vect                      = [1:N , 5001:5001+N-1];
  indextrain                = vect(randperm(length(vect)));
  indextest                 = (1:length(y));
  indextest(indextrain)     = [];

  ytrain                    = y(indextrain);
  ytest                     = y(indextest);




  tic,options.param         = fast_haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);,toc
  [ytest_est , fxtest]      = haar_adaboost_binary_predict_cascade(II(: , : , indextest) , options);

  indp                      = find(ytest == 1);
  indn                      = find(ytest ==-1);

  tp                        = sum(ytest_est(indp) == ytest(indp))/length(indp)
  fp                        = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn)
  perf                      = sum(ytest_est == ytest)/length(ytest)

  [tpp , fpp , threshold]   = basicroc(ytest , fxtest);

  figure(1)
  plot(fpp , tpp , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])

  figure(2)
  plot(fxtest)



  figure(3)
  imagesc(I)
  hold on

  best_feats          = (options.F(: , options.param(1 , 1:nb_feats)));
  x                   = double(best_feats(2 , :)) + 0.5 ;	
  y                   = double(best_feats(3 , :)) + 0.5;
  w                   = best_feats(4 , :);
  h                   = best_feats(5 , :);
  indR                = fix(best_feats(6 , :) + 1)/10 + 1;
  R                   = options.rect_param(4 , indR);
  
    for f = 1 : nb_feats
     for r = 0:R(f)-1
 		
  		coeffw  = w(f)/options.rect_param(2 , indR(f) + r);		
  		coeffh  = h(f)/options.rect_param(3 , indR(f) + r);
  		xr      = (x(f) + double(coeffw*options.rect_param(6 , indR(f) + r)));
  		yr      = (y(f) + double(coeffh*options.rect_param(7 , indR(f) + r))) ;
  		wr      = double(coeffw*(options.rect_param(8 , indR(f) + r)  - 0));
  		hr      = double(coeffh*(options.rect_param(9 , indR(f) + r) - 0));
  		s       = options.rect_param(10 , indR(f) + r);
  		if (s == 1)
            
  			color   = [0.9 0.9 0.9];
          
          else
  
    	    color   = [0.1 0.1 0.1];
  
          end
  	    hh      = rectangle('Position', [xr,  yr ,  wr ,  hr] );
        p       = patch([xr , xr+wr , xr + wr , xr] , [yr , yr , yr + hr , yr + hr] , color);
  		alpha(p , 0.8);
  	    set(hh , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  
  	end
   end
  hold off
  title(sprintf('Best %d Haar features' , nb_feats) , 'fontsize' , 13)
  colormap(gray)



  Example 3
  ---------

  clear, close all
  load viola_24x24
  options                   = load('haar_dico_2.mat');
  II                        = image_integral_standard(X);
  %clear X
  y                         = int8(y);
  options.T                 = 10;

  [Ny , Nx , P]             = size(II);
  options.F                 = haar_featlist(Ny , Nx , options.rect_param);
  options.G                 = Haar_matG(Ny , Nx , options.rect_param);


  N                         = 2500;
  vect                      = [1:N , 5001:5001+N-1];
  indextrain                = vect(randperm(length(vect)));
  indextest                 = (1:length(y));
  indextest(indextrain)     = [];

  ytrain                    = y(indextrain);
  ytest                     = y(indextest);

  indp                      = find(ytest == 1);
  indn                      = find(ytest ==-1);

  options.fine_threshold    = 0;
  tic,options.param         = fast_haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);,toc
  [ytest_est , fxtest]      = haar_adaboost_binary_predict_cascade(II(: , : , indextest) , options);


  tp                        = sum(ytest_est(indp) == ytest(indp))/length(indp)
  fp                        = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn)
  perf                      = sum(ytest_est == ytest)/length(ytest)

  [tpp , fpp , threshold]   = basicroc(ytest , fxtest);


  options.fine_threshold    = 1;
  tic,options.param         = fast_haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);,toc
  [ytest_est1 , fxtest1]    = haar_adaboost_binary_predict_cascade(II(: , : , indextest) , options);

  tp1                       = sum(ytest_est1(indp) == ytest(indp))/length(indp)
  fp1                       = 1 - sum(ytest_est1(indn) == ytest(indn))/length(indn)
  perf1                     = sum(ytest_est1 == ytest)/length(ytest)

  [tpp1 , fpp1 , threshold1] = basicroc(ytest , fxtest1);


  figure(1)
  plot(fpp , tpp , fpp1 , tpp1 , 'r' , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])
  legend('Without fine threshold', 'With fine threshold' , 'location' , 'southeast');


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009


 Changelog :  - Add OpenMP support
 ----------   - Add fine_threshold option


 Reference : M-T Pham and T-J Cham, "Fast trainning and selection of Haar features using statistics in boosting-based face detection", 
 ---------  in Proc. 11th IEEE ICCV'07


*/

#include <math.h>
#include "mex.h"

#ifdef OMP 
 #include <omp.h>
#endif

#ifndef MAX_THREADS
#define MAX_THREADS     64
#endif

#define huge            1e300
#define verytiny        1e-15

#define tolx            10e-6
#define tolf            10e-6
#define maxfun          200
#define maxiter         200
#define rho             1
#define chi             2
#define psi             0.5
#define sigma           0.5
#define usual_delta     0.05            
#define zero_term_delta 0.00025
#define eps             2.22044604925031e-016


#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))
#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif



#if defined(__OS2__)  || defined(__WINDOWS__) || defined(WIN32) || defined(WIN64) || defined(_MSC_VER)
#define BLASCALL(f) f
#else
#define BLASCALL(f) f ##_
#endif

struct opts
{
  double    *G;
#ifdef OS64
  mwIndex   *irG;
  mwIndex   *jcG;
#else
  int       *irG;
  int       *jcG;
#endif
  int        nF;
  int        T;
  int        weaklearner;
  double    *premodel;
  int        Npremodel;
  int       fine_threshold;
#ifdef OMP 
    int     num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

extern void  BLASCALL(dsyrk)(char * , char * , int * , int * , double * , double *, int *, double * , double * , int *);
#ifdef OS64
double fast_haar_feat(double * , int , double * , mwIndex *, mwIndex *);
#else
double fast_haar_feat(double * , int , double * , int *, int *);
#endif
double erf(double );
double error_fcn (double , double , double , double , double , double , double);
double neldermead_error_fcn(double * , double , double , double , double , double , double );
void   fast_haar_adaboost_binary_train_cascade(double *, char * , int , int , int , struct opts , double * );

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{		
    double *II;	
	char *y;
#ifdef OS64
	const mwSize *dimsII;
#else
	const int *dimsII;
#endif
	double *param;
	int Ny , Nx , NyNx , N; 
	mxArray *mxtemp;
	struct opts options;
	double *tmp;
	int tempint;

	options.weaklearner    = 2;
	options.T              = 100;
	options.Npremodel      = 0;
	options.fine_threshold = 0;

#ifdef OMP 
    options.num_threads    = -1;
#endif

    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) ==3) && (!mxIsEmpty(prhs[0])) && (mxIsDouble(prhs[0])) )
	{
		II          = mxGetPr(prhs[0]);
		dimsII      = mxGetDimensions(prhs[0]);
		Ny          = dimsII[0];
		Nx          = dimsII[1];
		N           = dimsII[2];
		NyNx        = Ny*Nx;
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
	
	if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
	{	
		mxtemp                            = mxGetField(prhs[2] , 0 , "G");	
		if(mxtemp != NULL)
		{
			if(!mxIsSparse(mxtemp) || mxGetM(mxtemp) != NyNx)
			{
				mexErrMsgTxt("G must be sparse matrix (Ny*Nx x nF), see Haar_matG function to build it");   
			}
			options.G                     = mxGetPr(mxtemp);
			options.irG                   = mxGetIr(mxtemp);
			options.jcG                   = mxGetJc(mxtemp);
			options.nF                    = mxGetN(mxtemp);		
		}
		
		mxtemp                            = mxGetField(prhs[2] , 0 , "T");	
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) )
			{
				mexPrintf("T  > 0");				
				options.T                 = 100;
			}
			else
			{
				options.T                  = tempint;	
			}	
		}	
		
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

		mxtemp                            = mxGetField(prhs[2] , 0 , "premodel");	
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 4)
			{
				mexErrMsgTxt("premodel must be (4 x Npremodel) in double format");

			}
			options.premodel      =  mxGetPr(mxtemp);
			options.Npremodel     = mxGetN(mxtemp);
		}

		mxtemp                            = mxGetField( prhs[2] , 0, "fine_threshold" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 1))
			{								
				mexPrintf("fine_threshold must be {0,1}, force to 0");
				options.fine_threshold    = 0;
			}
			else
			{
				options.fine_threshold    = tempint;	
			}			
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
		
		fast_haar_adaboost_binary_train_cascade(II , y , Ny , Nx , N , options , param);	
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  fast_haar_adaboost_binary_train_cascade(double *II , char *y , int Ny , int Nx , int N , struct opts options , double *param )
{
	double *premodel = options.premodel , *G = options.G;

#ifdef OS64
  mwIndex   *irG = options.irG , *jcG = options.jcG;
#else
	int  *irG = options.irG , *jcG = options.jcG;
#endif
	int i , j , f , t , T = options.T , nF = options.nF ,Npremodel = options.Npremodel , fine_threshold = options.fine_threshold;	
#ifdef OMP 
	int num_threads = options.num_threads;
#endif
	int NyNx = Ny*Nx , indNyNx , ind1NyNx , ind2NyNx , indM  , N1 = 0, N2 = 0 , featuresIdx_opt , indGi , indGj;
	double cteN =1.0/(double)N  , Errormin , errm , fm , sumw , cm , wtemp  , temp1 , temp2 , th_opt , z;
	double *w , *my1 , *my2  , *Z1c , *Z2c , *cov1c , *cov2c;
	char *h;
	double   a_opt , p1  , p2 , invp1 , invp2 , sqrtw , one = 1.0 , zero = 0.0 , Gi , Gj , ctep , alpha , beta , gamma , delta , sqrtdelta;
	double x1 , x2 , Err;
	double  m1c , m2c , var1c , var2c , std1c , std2c;
	char uplo = 'L' , trans = 'N';
	int *indexF;
	
#ifdef OMP 
	num_threads          = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
	omp_set_num_threads(num_threads);
#endif

	w                    = (double *)malloc(N*sizeof(double));

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (y,w,N,cteN) reduction (+:N1,N2)
#endif		
	for(i = 0 ; i < N ; i++)
	{		
		w[i]             = cteN;
		if(y[i] == 1)
		{		
			N1++;	
		}
		else
		{
			N2++;	
		}	
	}
		
	Z1c                  = (double *)malloc(NyNx*N1*sizeof(double));
	Z2c                  = (double *)malloc(NyNx*N2*sizeof(double));
	cov1c                = (double *)malloc(NyNx*NyNx*sizeof(double));
	cov2c                = (double *)malloc(NyNx*NyNx*sizeof(double));
	my1                  = (double *)malloc(NyNx*sizeof(double));
	my2                  = (double *)malloc(NyNx*sizeof(double));
	h                    = (char *)malloc(N*sizeof(char));
	indexF               = (int *)malloc(nF*sizeof(int));
	

	for(i = 0 ; i < nF ; i++)
	{		
		indexF[i]        = i;
	}
	
	/* Previous premodel */
	
	indM                 = 0;
#ifdef OMP 
#pragma omp parallel for default(none) private(j,i,featuresIdx_opt,th_opt,a_opt,z,fm) shared(premodel,Npremodel,N,NyNx,II,G,irG,jcG,Ny,y,w) reduction (+:indM,sumw) 
#endif
	for(j = 0 ; j < Npremodel ; j++)
	{
		featuresIdx_opt  = ((int) premodel[0 + indM]) - 1;	
		th_opt           = premodel[1 + indM];
		a_opt            = premodel[2 + indM];
		
		sumw             = 0.0;		
		for (i = 0 ; i < N ; i++)
		{
			z            = fast_haar_feat(II + j*NyNx , featuresIdx_opt , G , irG , jcG);	
			fm           = (a_opt*sign(z - th_opt));
			w[i]        *= exp(-y[i]*fm);
			sumw        += w[i];
		}
			
		sumw            = 1.0/sumw;
		for (i = 0 ; i < N ; i++)
		{	
			w[i]         *= sumw;
		}
		indM            += 4;
	}
	
	indM  = 0;
	for(t = 0 ; t < T ; t++)
	{		
		for(j = 0 ; j < NyNx ; j++)
		{
			my1[j]  = 0.0;	
			my2[j]  = 0.0;		
		}

		p1       = 0.0;
		p2       = 0.0;

		for(i = 0 ; i < N ; i++)
		{
			if(y[i] == 1)
			{		
				wtemp   = w[i];			
				p1     += wtemp;

				for (j = 0 ; j < NyNx ; j++)
				{
					my1[j]    += wtemp*II[j + i*NyNx];		
				}
			}
			else
			{			
				wtemp   = w[i];
				p2     += wtemp;
				for (j = 0 ; j < NyNx ; j++)
				{				
					my2[j]    += wtemp*II[j + i*NyNx];	
				}		
			}			
		}

		invp1       = 1.0/p1;	
		invp2       = 1.0/p2;
		for(j = 0 ; j < NyNx ; j++)
		{	
			my1[j]  *= invp1;	
			my2[j]  *= invp2;
		}

		ind1NyNx    = 0;
		ind2NyNx    = 0;
		for(i = 0 ; i < N ; i++)
		{	
			if(y[i] == 1)
			{			
				sqrtw                   = sqrt(w[i]*invp1);	
				for( j = 0 ; j < NyNx ; j++)
				{			
					Z1c[j + ind1NyNx]   = sqrtw*(II[j + i*NyNx] - my1[j]);	
				}
				ind1NyNx               += NyNx;
			}
			else
			{			
				sqrtw                   = sqrt(w[i]*invp2);	
				for( j = 0 ; j < NyNx ; j++)
				{
					Z2c[j + ind2NyNx]   = sqrtw*(II[j + i*NyNx] - my2[j]);		
				}
				ind2NyNx               += NyNx;	
			}			
		}

		BLASCALL(dsyrk) (&uplo , &trans , &NyNx , &N1 , &one , Z1c , &NyNx , &zero , cov1c , &NyNx);
		BLASCALL(dsyrk) (&uplo , &trans , &NyNx , &N2 , &one , Z2c , &NyNx , &zero , cov2c , &NyNx);

		indNyNx        = 0;
		for (i = 0 ; i < NyNx - 1; i++)
		{
			for (j = i + 1 ; j < NyNx ; j++)		
			{
				cov1c[i + j*NyNx] = cov1c[j + indNyNx];	
				cov2c[i + j*NyNx] = cov2c[j + indNyNx];			
			}
			indNyNx   += NyNx;
		}
		ctep                          = p1/p2;
		Errormin                      = huge;	

#ifdef OMP 
#pragma omp parallel for default(none) private(f,i,j,m1c,m2c,Gi,indGi,var1c,var2c,temp1,temp2,Gj,indGj,indNyNx,alpha,beta,gamma,delta,sqrtdelta,std1c,std2c,x1,x2,Err) \
	shared (Errormin,featuresIdx_opt,th_opt,a_opt,indexF,irG,jcG,my1,my2,cov1c,cov2c,G,N,NyNx,p1,p2,ctep,nF,fine_threshold) 
#endif

		for (f = 0 ; f < nF ; f++)
		{
			if(indexF[f] != -1)
			{
				m1c                   = 0.0;	
				m2c                   = 0.0;
				for( i = jcG[f] ; i < jcG[f + 1] ; i++)
				{
					Gi                = G[i];		
					indGi             = irG[i];
					m1c              += my1[indGi]*Gi;
					m2c              += my2[indGi]*Gi;
				}

				var1c                 = 0.0;
				var2c                 = 0.0;
				for(j = jcG[f] ; j < jcG[f + 1] ; j++)
				{

					temp1             = 0.0;			
					temp2             = 0.0;
					Gj                = G[j];
					indGj             = irG[j];
					indNyNx           = indGj*NyNx;
					for( i = jcG[f] ; i < jcG[f + 1] ; i++)
					{			
						Gi             = G[i];	
						indGi          = irG[i] + indNyNx;
						temp1         += cov1c[indGi]*Gi;
						temp2         += cov2c[indGi]*Gi;
					}
					var1c             += temp1*Gj;
					var2c             += temp2*Gj;			
				}

				alpha                  = var1c - var2c;	
				beta                   = (m1c*var2c - m2c*var1c);
				gamma                  = m2c*m2c*var1c - m1c*m1c*var2c  +  2.0*var1c*var2c*log(ctep*(var2c/var1c));
				delta                  = beta*beta - alpha*gamma;

				if(delta < 0)
				{
					delta              = -delta;	
				}

				sqrtdelta              = sqrt(delta);
				std1c                  = 1.0/sqrt(var1c); /* 1.0/sqrt(2.0*var1c); */
				std2c                  = 1.0/sqrt(var2c); /* 1.0/sqrt(2.0*var2c); */
				if(alpha != 0.0)
				{
					x1                 = (-beta + sqrtdelta)/(alpha);	
					x2                 = (-beta - sqrtdelta)/(alpha);
				}
				else
				{	
					x1                 = -gamma/beta;	
					x2                 = x1;
				}

/*
				if(m1c > m2c)
				{	
					Err1               = 0.5*(p2*(1.0 - erf((x1 - m2c)*std2c)) + p1*(1.0 + erf((x1 - m1c)*std1c)));	
					Err2               = 0.5*(p2*(1.0 - erf((x2 - m2c)*std2c)) + p1*(1.0 + erf((x2 - m1c)*std1c)));		
				}
				else
				{
					Err1               = 0.5*(p1*(1.0 - erf((x1 - m1c)*std1c)) + p2*(1.0 + erf((x1 - m2c)*std2c)));	
					Err2               = 0.5*(p1*(1.0 - erf((x2 - m1c)*std1c)) + p2*(1.0 + erf((x2 - m2c)*std2c)));		
				}

				if(Err1 < Err2)	
				{
					if(Err1 < Errormin)
					{
						Errormin           = Err1;	
						featuresIdx_opt    = f;
						th_opt             = x1;
						a_opt              = -1.0;
					}
				}
				else
				{
					if(Err2 < Errormin)
					{		
						Errormin           = Err2;	
						featuresIdx_opt    = f;
						th_opt             = x2;
						a_opt              = 1.0;
					}
				}
*/

				if(m1c > m2c)
				{
					if(fine_threshold)
					{
						Err                = neldermead_error_fcn(&x1,p1,p2,m1c,m2c,std1c,std2c);
					}
					else
					{
						Err                = error_fcn(x1,p1,p2,m1c,m2c,std1c,std2c);
					}
					if(Err < Errormin)
					{
						Errormin           = Err;
						featuresIdx_opt    = f;
						th_opt             = x1;
						a_opt              = 1.0;
					}
				}
				else
				{
					if(fine_threshold)
					{
						Err                = neldermead_error_fcn(&x2,p2,p1,m2c,m1c,std2c,std1c);
					}
					else
					{
						Err                = error_fcn(x2,p2,p1,m2c,m1c,std2c,std1c);
					}
					if(Err < Errormin)
					{
						Errormin           = Err;
						featuresIdx_opt    = f;
						th_opt             = x2;
						a_opt              = -1.0;
					}
				}
			}
		}

		errm             = 0.0;
#ifdef OMP 
#pragma omp parallel for default(none) private(i,z) shared (II,y,w,G,irG,jcG,h,N,NyNx,a_opt,th_opt,featuresIdx_opt) reduction (+:errm)
#endif

		for (i = 0 ; i < N ; i++)
		{
			z            = fast_haar_feat(II + i*NyNx , featuresIdx_opt , G , irG , jcG);	  
			h[i]         = (char) (a_opt*sign(z - th_opt));
			if(y[i] != h[i])
			{
				errm    += w[i];	  
			}
		}

		cm              = 0.5*log((1.0 - errm)/errm);
		sumw            = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (w,y,h,N,cm) reduction (+:sumw)
#endif
		for (i = 0 ; i < N ; i++)
		{
			w[i]        *= exp(-(y[i]*h[i])*cm);	  
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
		param[1 + indM]         = th_opt;
		param[2 + indM]         = a_opt*cm;
		param[3 + indM]         = 0.0;
		indM                   += 4;
	}

	free(w);
	free(h);
	free(my1);
	free(my2);
	free(Z1c);
	free(Z2c);
	free(cov1c);
	free(cov2c);
	free(indexF);	
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
double error_fcn (double x , double p1 , double p2 , double m1c , double m2c , double std1c , double std2c)
{
	return ( 0.5*(1.0 + p1*(erf((x - m1c)*std1c)) - p2*(erf((x - m2c)*std2c))) );
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double neldermead_error_fcn(double *x , double p1 , double p2, double m1, double m2, double std1, double std2)
{
	double v0 , v1 , fv0 , fv1 , temp , tolfv0 = 10.0*eps , tolv0 = 10.0*eps;
	double xbar , xr , fxr , xc , fxc , xcc , fxcc , xe , fxe;  
	int how , itercount  = 0, func_evals;

	v0              = (*x); 
	fv0             = error_fcn(v0 , p1 , p2 , m1 , m2 , std1 , std2);

	func_evals      = 1;

	v1              = v0;
	if (v1 != 0.0)
	{
		v1          = (1.0 + usual_delta)*v1;
	}
	else
	{
		v1          = zero_term_delta;
	}
	fv1             = error_fcn(v1 , p1 , p2 , m1 , m2 , std1 , std2);
	if(fv1 < fv0)
	{
		temp        = fv1;
		fv1         = fv0;
		fv0         = temp;
		temp        = v1;
		v1          = v0;
		v0          = temp;
	}

	how             = 0;
	itercount++;
	func_evals++;

	while ((func_evals < maxfun) && (itercount < maxiter))
	{
		if ( (abs(fv0-fv1) <= max(tolf,tolfv0*fv0)) && (abs(v1-v0) <= max(tolx,tolv0*v0)) )
		{
			break;
		}
		xbar       = v0;
		xr         = (1.0 + rho)*xbar - rho*v1;
		fxr        = error_fcn(xr , p1 , p2 , m1 , m2 , std1 , std2);
		func_evals++;

		if (fxr < fv0)
		{
			xe         = (1.0 + rho*chi)*xbar - rho*chi*v1;
			fxe        = error_fcn(xe , p1 , p2 , m1 , m2 , std1 , std2);
			func_evals++;
			if (fxe < fxr)
			{
				v1  = xe;
				fv1 = fxe;
				how = 1;
			}
			else
			{
				v1  = xr;
				fv1 = fxr;
				how = 2;
			}
		}
		else
		{
			if (fxr < fv1)
			{
				xc         = (1.0 + psi*rho)*xbar - psi*rho*v1;
				fxc        = error_fcn(xc , p1 , p2 , m1 , m2 , std1 , std2);
				func_evals++;
				if (fxc <= fxr)
				{
					v1     = xc;
					fv1    = fxc;
					how    = 3;
				}
				else
				{
					how    = 4;
				}
			}
			else
			{
				xcc         = (1.0 - psi)*xbar + psi*v1;
				fxcc        = error_fcn(xcc , p1 , p2 , m1 , m2 , std1 , std2);
				func_evals++;       
				if (fxcc < fv1)
				{
					v1     = xcc;
					fv1    = fxcc;
					how    = 3;
				}
				else
				{
					how    = 4;
				}
			}
			if (how == 4)
			{
				v1         = v0 + sigma*(v1 - v0);
				fv1        = error_fcn(v1 , p1 , p2 , m1 , m2 , std1 , std2);
				func_evals++;
			}
		}

		if(fv1 < fv0)
		{
			temp        = fv1;
			fv1         = fv0;
			fv0         = temp;
			temp        = v1;
			v1          = v0;
			v0          = temp;
		}
		
		itercount++;
	}
(*x)  = v0;
return fv0;

}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
#ifdef OS64
double fast_haar_feat(double *II , int featidx , double *G , mwIndex *irG , mwIndex *jcG)
#else
double fast_haar_feat(double *II , int featidx , double *G , int *irG , int *jcG)
#endif
{
	int i;
	double val = 0;

	for(i = jcG[featidx] ; i < jcG[featidx + 1] ; i++)
	{
		val += G[i]*II[irG[i]];
	}
	return val;		
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double erf(double x)
{
	const double p1[5] = 
	{
		3.20937758913846947e03,
			3.77485237685302021e02,
			1.13864154151050156e02,
			3.16112374387056560e00,
			1.85777706184603153e-1
	};
	const double q1[4] = 
	{
		2.84423683343917062e03,
			1.28261652607737228e03,
			2.44024637934444173e02,
			2.36012909523441209e01
	};
	const double p2[9] = 
	{ 
		1.23033935479799725e03,
			2.05107837782607147e03,
			1.71204761263407058e03,
			8.81952221241769090e02,
			2.98635138197400131e02,
			6.61191906371416295e01,
			8.88314979438837594e00,
			5.64188496988670089e-1,
			2.15311535474403846e-8
	};
	const double q2[8] = 
	{ 
		1.23033935480374942e03,
			3.43936767414372164e03,
			4.36261909014324716e03,
			3.29079923573345963e03,
			1.62138957456669019e03,
			5.37181101862009858e02,
			1.17693950891312499e02,
			1.57449261107098347e01
	};
	const double p3[6] = 
	{
		6.58749161529837803e-4,
			1.60837851487422766e-2,
			1.25781726111229246e-1,
			3.60344899949804439e-1,
			3.05326634961232344e-1,
			1.63153871373020978e-2
	};
	const double q3[5] = 
	{ 
		2.33520497626869185e-3,
			6.05183413124413191e-2,
			5.27905102951428412e-1,
			1.87295284992346047e00,
			2.56852019228982242e00
	};
	
	int i;
	double xval, xx, p, q;
	bool NegativeValue;
	
    xval = x;
	
	if (xval<0) 	
	{
		xval=-xval; 
		NegativeValue=true;
	}
	else 
	{	
		NegativeValue=false;
	}
	if (xval<=0.46875)
    {
		xx = xval*xval;	
		p  = p1[4];
		q  = 1.0;
		for (i = 3 ; i>=0 ; i--) 
		{
			p = p*xx + p1[i]; 
			q = q*xx + q1[i];
		}
		xx=p/q;
		return(x*xx);
    }
	else if (xval<=4)	
	{
		xx = xval;
		p  = p2[8];
		q  = 1.0;
		
		for (i=7 ; i>=0 ; i--) 
		{
			p = p*xx + p2[i]; 	
			q = q*xx + q2[i];
		}
		
		xx = p/q;
		xx = exp(-xval*xval)*xx;
		if (NegativeValue) 
		{ 
			return((xx - 0.5) - 0.5);
		} 
		else 
		{	
			return((0.5 - xx) + 0.5);
		}
    }  
    else if (xval<10)
    {
		xx = 1.0/(xval*xval);	
		p  = p3[5];
		q  = 1.0;
		for (i = 4 ; i >= 0 ; i--) 
		{
			p = p*xx + p3[i]; 	
			q = q*xx + q3[i];
		}
		xx = p/q;
		xx = exp(-xval*xval)*(0.7071067811865475 - xx)/(xval);
		
		if (mxIsNaN(xx)) 	
		{
			xx        = 0.0;	
		}
		if (NegativeValue) 
		{	
			return((xx - 0.5) - 0.5); 	
		}
		else 	
		{
			return((0.5 - xx) + 0.5);	
		}
    }
    else
	{
		if (NegativeValue) 
		{		
			return(-1); 
		}
		else 	
		{
			return(1);
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
