
/*

  Train Haar features with adaboost classifier

  Usage
  ------
  
  param = haar_adaboost_binary_train_cascade(II , y , [options]);

  
  Inputs
  -------

  II                                    Images Integral (Ny x Nx x N) standardized in DOUBLE format
  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
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
           F                            Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle
           T                            Number of weak learners (default T = 100)
           weaklearner                  Choice of the weak learner used in the training phase (default weaklearner = 2)
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
           premodel                     Classifier's premodels parameter up to n-1 stage (4 x Npremodels)(default premodel = [] for stage n=1)

If compiled with the "OMP" compilation flag
	     num_threads                    Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  param                                 Parameter output matrix (4 x T) for current stage n of the classifier's premodel
        featureIdx                      Feature indexes of the T best weaklearners (1 x T)
		th                              Optimal Threshold parameters (1 x T)
		a                               WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)
		b                               Zeros (1 x T), i.e. b = zeros(1 , T)


  To compile
  ----------

  mex  -output  haar_adaboost_binary_train_cascade.dll  haar_adaboost_binary_train_cascade.c

  mex  -f mexopts_intel10.bat -output  haar_adaboost_binary_train_cascade.dll  haar_adaboost_binary_train_cascade.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output  haar_adaboost_binary_train_cascade.dll  haar_adaboost_binary_train_cascade.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

  Example 1
  ---------

  clear, close all
  load viola_24x24
  y                         = int8(y);
  options                   = load('haar_dico_5.mat');

  II                        = image_integral_standard(X);
  clear X
  [Ny , Nx , P]             = size(II);

  options.T                 = 3;
  options.F                 = haar_featlist(Ny , Nx , options.rect_param);
  
  N                         = 2000;
  vect                      = [1:N , 5001:5001+N-1];
  indextrain                = vect(randperm(length(vect)));
  indextest                 = (1:length(y));
  indextest(indextrain)     = [];

  ytrain                    = y(indextrain);
  ytest                     = y(indextest);
  options.param             = haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);
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


  [dum , ind]              = sort(ytest , 'descend');
  figure(2)
  plot(fxtest(ind))


  figure(3)
  plot(abs(options.param(3 , :)) , 'linewidth' , 2)
  grid on
  xlabel('Weaklearner m')
  ylabel('|a_m|')


  Example 2
  ---------

  clear, close all 
  load viola_24x24
  options            = load('haar_dico_2.mat');

  II                 = image_integral_standard(X);
  [Ny , Nx , P]      = size(II);
  Nimage             = 110;
  nb_feats           = 5;


  options.T          = 5;
  options.F          = haar_featlist(Ny , Nx , options.rect_param);

  index              = randperm(length(y));
  
  y                  = int8(y);
  tic,otpions.param  = haar_adaboost_binary_train_cascade(II(: , : , index) , y(index),  options);,toc
  [yest , fx]        = haar_adaboost_binary_predict_cascade(II , options);
  indp               = find(y == 1);
  indn               = find(y ==-1);

  tp                 = sum(yest(indp) == y(indp))/length(indp)
  fp                 = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf               = sum(yest == y)/length(y)

  [tpp , fpp]        = basicroc(y , fx);

  figure(1)
  plot(fpp , tpp)
  axis([-0.02 , 1.02 , -0.02 , 1.02])


  [dum , ind]        = sort(y , 'descend');
  figure(2)
  plot(fx(ind))


  I                  = X(: , : , Nimage);


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
  title(sprintf('Best %d Haar features with Adaboosting' , nb_feats) , 'fontsize' , 13)
  colormap(gray)


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Changelog :  - Add OpenMP support
 ----------   


 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999

              [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
			      ICB07

			  [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
			  [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004


*/

#include <math.h>
#include "mex.h"

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
	double        *rect_param;
	int            nR;
	unsigned int  *F;
	int            nF;
	int            T;
    int            weaklearner;
    double        *premodel;
    int            Npremodel;
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
void  adaboost_decision_stump(double *, char *, int , int , int , struct opts ,  double *);

/*---------------------------------------------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    double *II;
	char *y;
	const int *dimsII;	
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	double *param;
	int i , Ny , Nx , N; 
	mxArray *mxtemp;
	struct opts options;	
	double *tmp;	
	int tempint;
	
	options.T           = 100;
	options.Npremodel   = 0;
	options.nR          = 4;
    options.nF          = 0;
	options.weaklearner = 2; 
#ifdef OMP 
    options.num_threads = -1;
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


	if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
	{
		mxtemp                             = mxGetField( prhs[2] , 0, "rect_param" );
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

		mxtemp                             = mxGetField( prhs[2] , 0, "F" );
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

		mxtemp                            = mxGetField( prhs[2] , 0, "T" );
		if(mxtemp != NULL)
		{

			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];

			if((tempint < 0))
			{
				mexPrintf("T > 0, force to 100");		
				options.T                 = 100;
			}
			else
			{
				options.T                 = tempint;	
			}			
		}

		mxtemp                            = mxGetField( prhs[2] , 0, "weaklearner" );
		if(mxtemp != NULL)
		{

			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];

			if((tempint < 0) || (tempint > 3))
			{
				mexPrintf("weaklearner = {0,1,2}, force to 2");		
				options.weaklearner       = 2;
			}
			else
			{
				options.weaklearner       = tempint;	
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
	}   

		/*------------------------ Main Call ----------------------------*/

	if(options.weaklearner == 2)
	{	
		plhs[0]                        = mxCreateNumericMatrix(4 , options.T , mxDOUBLE_CLASS,mxREAL);
		param                          = mxGetPr(plhs[0]);
		
		adaboost_decision_stump(II , y  , Ny , Nx , N , options , param);
	}

   /*--------------------------- Free memory -----------------------*/

	if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
	{
		if ( mxGetField( prhs[2] , 0 , "rect_param" ) == NULL )	
		{
			mxFree(options.rect_param);
		}

		if ( mxGetField( prhs[2] , 0 , "F" ) == NULL )	
		{
			mxFree(options.F);
		}
	}
	else
	{
		mxFree(options.rect_param);
		mxFree(options.F);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  adaboost_decision_stump(double *II , char *y , int Ny , int Nx , int N  , struct opts options ,  double *param)
{
	double *rect_param = options.rect_param , *premodel = options.premodel;
	unsigned int *F = options.F;
	int T = options.T , Npremodel = options.Npremodel , nR = options.nR , nF = options.nF;
#ifdef OMP 
	int num_threads = options.num_threads;
#endif
	int i , j , t;
	int NyNx = Ny*Nx , indM  , ind , N1 = N - 1 , featuresIdx_opt;
	double cteN =1.0/(double)N  , Tplus , Tminus , Splus , Sminus , Errormin , errm , fm , sumw , cm , Errplus , Errminus , wtemp , a_opt ;
	double *w;
	char ytemp;
	double *xtemp , z , th_opt ;
	char  *h;
	int *index;
	int *indexF;

	w                   = (double *)malloc(N*sizeof(double));
	h                   = (char *)malloc(N*sizeof(char));
	indexF              = (int *)malloc(nF*sizeof(int));
#ifdef OMP 

#else
	xtemp               = (double *)malloc(N*sizeof(double ));
	index               = (int *)malloc(N*sizeof(int));
#endif

#ifdef OMP 
	num_threads          = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
	omp_set_num_threads(num_threads);
#endif

	for(i = 0 ; i < N ; i++)
	{	
		w[i]            = cteN;	
	}
	for(i = 0 ; i < nF ; i++)
	{		
		indexF[i]       = i;
	}

	/* Previous premodel */

	indM                 = 0;

#ifdef OMP 
#pragma omp parallel for default(none) private(j,i,featuresIdx_opt,th_opt,a_opt,z,fm) shared(premodel,Npremodel,N,NyNx,II,rect_param,F,Ny,nR,nF,y,w) reduction (+:indM,sumw) 
#endif
	for(j = 0 ; j < Npremodel ; j++)
	{
		featuresIdx_opt  = ((int) premodel[0 + indM]) - 1;
		th_opt           = premodel[1 + indM];
		a_opt            = premodel[2 + indM];
		sumw             = 0.0;
		for (i = 0 ; i < N ; i++)
		{
			z            = haar_feat(II + j*NyNx , featuresIdx_opt , rect_param , F , Ny , nR , nF);
			fm           = a_opt*sign(z - th_opt);
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

	indM  = 0;
	for(t = 0 ; t < T ; t++)
	{		

		Tplus            = 0.0;
		Tminus           = 0.0;
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
#pragma omp parallel  default(none) private(xtemp,index,ytemp,wtemp,j,i,ind,Errplus,Errminus) shared(N,N1,NyNx,Ny,nR,nF,indexF,II,w,y,rect_param,F,featuresIdx_opt,th_opt,a_opt,Errormin,Tplus,Tminus) reduction (+:Splus,Sminus) 
#endif
		{

#ifdef OMP 
			xtemp               = (double *)malloc(N*sizeof(double ));
			index               = (int *)malloc(N*sizeof(int));
#else
#endif

#ifdef OMP 
#pragma omp for nowait
#endif
			for(j = 0 ; j < nF  ; j++)	
			{
				if(indexF[j] != -1)
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
						wtemp       = w[ind];
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
#ifdef OMP
			free(index);
			free(xtemp);
#else

#endif
		}

		errm             = 0.0;		

#ifdef OMP 
#pragma omp parallel for default(none) private(i,z) shared (II,w,y,h,a_opt,th_opt,featuresIdx_opt,rect_param,F,N,NyNx,Ny,nR,nF) reduction (+:errm) 
#endif
		for (i = 0 ; i < N ; i++)
		{	
			z            = haar_feat(II + i*NyNx , featuresIdx_opt , rect_param , F , Ny , nR , nF);
			h[i]         = a_opt*sign(z - th_opt);

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

		indexF[featuresIdx_opt]   = -1;
		param[0 + indM]           = (double) (featuresIdx_opt + 1);
		param[1 + indM]           = th_opt;
		param[2 + indM]           = a_opt*cm;		
		param[3 + indM]           = 0.0;
		indM                     += 4;
	}

	free(w);
	free(h);
	free(indexF);
#ifdef OMP

#else
	free(index);
	free(xtemp);

#endif
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
	int Y , X ;
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
