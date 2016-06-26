
/*

  Train Multiblock Local Binary Pattern with gentleboosting classifier for binary problem

  Usage
  ------

  param   = mblbp_adaboost_binary_train_cascade(X , y , [options]);

  
  Inputs
  -------

  X                                     Features matrix (d x N) (or (N x d) if transpose = 1) in UINT8 format in UINT8 format
  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
  options
         T                              Number of weak learners (default T = 100)	 
	     weaklearner                    Choice of the weak learner used in the training phase (default weaklearner = 2)
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
         premodel                       Classifier's premodels parameter up to n-1 stage (4 x Npremodels)(default premodel = [] for stage n=1)
		 transpose                      Suppose X' as input (in order to speed up Boosting algorithm avoiding internal transposing, default tranpose = 0)
  If OMP directive is added, OpenMP support for multicore computation
	    num_threads                     Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------  
  param                                 param output (4 x T) for current stage n of the classifier's premodel
       featureIdx                       Feature indexes of the T best weaklearners (1 x T)
	   th                               Optimal Threshold parameters (1 x T)
	   a                                WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)
	   b                                Zeros (1 x T), i.e. b = zeros(1 , T)


  To compile
  ----------


  mex  -output mblbp_adaboost_binary_train_cascade.dll mblbp_adaboost_binary_train_cascade.c

  mex  -f mexopts_intel10.bat -output mblbp_adaboost_binary_train_cascade.dll mblbp_adaboost_binary_train_cascade.c

  If OMP directive is added, OpenMP support for multicore computation

  mex  -v -DOMP -f mexopts_intel10.bat -output mblbp_adaboost_binary_train_cascade.dll mblbp_adaboost_binary_train_cascade.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

  Example 1
  ---------

  clear, close all
  load viola_24x24
  y                     = int8(y);
  Ny                    = 24;
  Nx                    = 24;
  N                     = 8;
  Nimage                = 110;
  nb_feats              = 3;
  options.T             = 10;
  options.F             = mblbp_featlist(Ny , Nx);
  options.transpose     = 0;
  mapping               = getmapping(N,'u2');
  options.map           = uint8(mapping.table);
  options.map           = uint8(0:255);
  H                     = mblbp(X , options);

  figure(1)
  imagesc(H)
  title('MBLBP Features')
  drawnow


  index                 = randperm(length(y));

  if(options.transpose)
    tic,options.param   = mblbp_adaboost_binary_train_cascade(H(index , :) , y(index) , options);,toc
  else
    tic,options.param   = mblbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);,toc
  end
  [yest , fx]           = mblbp_adaboost_binary_predict_cascade(H , options);
  indp                  = find(y == 1);
  indn                  = find(y ==-1);

  tp                    = sum(yest(indp) == y(indp))/length(indp)
  fp                    = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf                  = sum(yest == y)/length(y)

  [tpp , fpp]           = basicroc(y , fx);

  figure(2)
  plot(fpp , tpp , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])


  [dum , ind]           = sort(y , 'descend');
  
  figure(3)
  plot(fx(ind))

  figure(4)

  best_feats = double(options.F(: , options.param(1 , 1:nb_feats)));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats(2,i)-best_feats(4,i) + 0.5,  best_feats(3,i)-best_feats(5,i) + 0.5 ,  3*best_feats(4,i) ,  3*best_feats(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d MBLBP features with Adaboosting' , nb_feats) , 'fontsize' , 13)
  colormap(gray)

  figure(5)
  plot(abs(options.param(3 , :)) , 'linewidth' , 2)
  grid on
  xlabel('Weaklearner m')
  ylabel('|a_m|')

  Example 2
  ---------

  clear, close all,drawnow
  load viola_24x24

  y             = int8(y);

  indp          = find(y == 1);
  indn          = find(y ==-1);


  Ny            = 24;
  Nx            = 24;
  N             = 8;
  T             = 20;
  Nimage        = 110;
  nb_feats      = 3;

  F             = mblbp_featlist(Ny , Nx);
  mapping       = getmapping(N,'u2');
  map0          = uint8(0:255);
  map1          = uint8(mapping.table);



  H0            = mblbp(X , F , map0 );
  H1            = mblbp(X , F , map1 );


  index         = randperm(length(y));


  tic,param0    = mblbp_adaboost_binary_train_cascade(H0(: , index) , y(index) , T);,toc
  [yest0 , fx0] = mblbp_adaboost_binary_predict_cascade(H0 , param0);

  tp0           = sum(yest0(indp) == y(indp))/length(indp)
  fp0           = 1 - sum(yest0(indn) == y(indn))/length(indn)
  perf0         = sum(yest0 == y)/length(y)

  [tpp0 , fpp0] = basicroc(y , fx0);



  tic,param1    = mblbp_adaboost_binary_train_cascade(H1(: , index) , y(index) , T);,toc
  [yest1 , fx1] = mblbp_adaboost_binary_predict_cascade(H1 , param1);

  tp1           = sum(yest1(indp) == y(indp))/length(indp)
  fp1           = 1 - sum(yest1(indn) == y(indn))/length(indn)
  perf1         = sum(yest1 == y)/length(y)

  [tpp1 , fpp1] = basicroc(y , fx1);


  figure(1)
  plot(fpp0 , tpp0 , fpp1 , tpp1 , 'r' , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])
  legend('MBLBP' , 'MBLBP_{u}')


  [dum , ind]        = sort(y , 'descend');
  
  figure(2)
  plot((1:length(ind)) , fx0(ind) , (1:length(ind)) , fx1(ind) , 'r')

  figure(3)
  best_feats0 = double(F(: , param0(1 , 1:nb_feats)));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats0(2,i)-best_feats0(4,i) + 0.5,  best_feats0(3,i)-best_feats0(5,i) + 0.5 ,  3*best_feats0(4,i) ,  3*best_feats0(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d MBLBP features' , nb_feats) , 'fontsize' , 13)
  colormap(gray)

  figure(4)

  best_feats1 = double(F(: , param1(1 , 1:nb_feats)));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats1(2,i)-best_feats1(4,i) + 0.5,  best_feats1(3,i)-best_feats1(5,i) + 0.5 ,  3*best_feats1(4,i) ,  3*best_feats1(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d MBLBP_{u} features' , nb_feats) , 'fontsize' , 13)
  colormap(gray)

  Example 3
  ---------

  clear, close all,drawnow
  load viola_24x24

  y             = int8(y);

  indp          = find(y == 1);
  indn          = find(y ==-1);


  Ny            = 24;
  Nx            = 24;
  N             = 8;
  T             = 15;
  Nimage        = 110;
  nb_feats      = 3;

  options.cs_opt = 1;



  F             = mblbp_featlist(Ny , Nx);
  mapping       = getmapping(N,'u2');
  map0          = uint8(mapping.table);
%  map0          = uint8(0:255);

  mapping       = getmapping(N/2,'u2');
  map1          = uint8(mapping.table);
 % map1          = uint8(0:15);


  H0            = mblbp(X , F , map0 );
  H1            = mblbp(X , F , map1 , options);


  index         = randperm(length(y));


  tic,param0    = mblbp_adaboost_binary_train_cascade(H0(: , index) , y(index) , T);,toc
  [yest0 , fx0] = mblbp_adaboost_binary_predict_cascade(H0 , param0);

  tp0           = sum(yest0(indp) == y(indp))/length(indp)
  fp0           = 1 - sum(yest0(indn) == y(indn))/length(indn)
  perf0         = sum(yest0 == y)/length(y)

  [tpp0 , fpp0] = basicroc(y , fx0);



  tic,param1    = mblbp_adaboost_binary_train_cascade(H1(: , index) , y(index) , T);,toc
  [yest1 , fx1] = mblbp_adaboost_binary_predict_cascade(H1 , param1);

  tp1           = sum(yest1(indp) == y(indp))/length(indp)
  fp1           = 1 - sum(yest1(indn) == y(indn))/length(indn)
  perf1         = sum(yest1 == y)/length(y)

  [tpp1 , fpp1] = basicroc(y , fx1);


  figure(1)
  plot(fpp0 , tpp0 , fpp1 , tpp1 , 'r' , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])
  legend('MBLBP' , 'CSMBLBP')


  [dum , ind]        = sort(y , 'descend');
  
  figure(2)
  plot((1:length(ind)) , fx0(ind) , (1:length(ind)) , fx1(ind) , 'r')

  figure(3)
  best_feats0 = double(F(: , param0(1 , 1:nb_feats)));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats0(2,i)-best_feats0(4,i) + 0.5,  best_feats0(3,i)-best_feats0(5,i) + 0.5 ,  3*best_feats0(4,i) ,  3*best_feats0(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d MBLBP features' , nb_feats) , 'fontsize' , 13)
  colormap(gray)

  figure(4)

  best_feats1 = double(F(: , param1(1 , 1:nb_feats)));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats1(2,i)-best_feats1(4,i) + 0.5,  best_feats1(3,i)-best_feats1(5,i) + 0.5 ,  3*best_feats1(4,i) ,  3*best_feats1(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d CSMBLBP features' , nb_feats) , 'fontsize' , 13)
  colormap(gray)



 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Changelog :  - Add OpenMP support
 ----------   - Add transpose option


 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999

              [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
			      ICB07

			  [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
			  [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004

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
	int            T;
    int            weaklearner;
    double        *premodel;
    int            Npremodel;
	int            transpose;
#ifdef OMP 
    int   num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void qsindex( unsigned char * , int * , int , int  );
void transposeX(unsigned char *, unsigned char * , int , int);
void adaboost_decision_stump(unsigned char *, char *, int , int ,  struct opts , double *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *X ;
	char *y;
	double *param;
	int d , N; 
	mxArray *mxtemp;
	struct opts options ;	
	double *tmp;
	int tempint;

	options.Npremodel   = 0;
	options.weaklearner = 2;
	options.transpose   = 0;

#ifdef OMP 
    options.num_threads = -1;
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

	if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
	{
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

		mxtemp                            = mxGetField( prhs[2] , 0, "transpose" );
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
		if(options.transpose)
		{
			adaboost_decision_stump(X , y  , N , d, options, param);	
		}
		else
		{
			adaboost_decision_stump(X , y  , d , N, options, param);	
		}
	}
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  adaboost_decision_stump(unsigned char *X , char *y , int d , int N ,  struct opts options , double *param )								 								 
{
	double *premodel = options.premodel;
	int T  = options.T, Npremodel = options.Npremodel , transpose = options.transpose; 
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	double cteN =1.0/(double)N;
	int i , j , t;
	int indN , Nd = N*d , ind , N1 = N - 1 , featuresIdx_opt;
	int indM , indice ;

	double *w;
	unsigned char *Xt, *xtemp;
	int *idX;
	double  sumw , fm  , a_opt;
	double Tplus , Tminus , Splus , Sminus , Errormin , cm , Errplus , Errminus;
	double *wtemp , errm;
	unsigned char th_opt ;
	char *ytemp ;
	char *h;
	int *indexF;

	idX              = (int *)malloc(Nd*sizeof(int));
	Xt               = (unsigned char *)malloc(Nd*sizeof(unsigned char ));
	w                = (double *)malloc(N*sizeof(double));
	h                = (char *)malloc(N*sizeof(char));
	indexF           = (int *)malloc(d*sizeof(int));

#ifdef OMP 

#else
	wtemp            = (double *)malloc(N*sizeof(double));
	xtemp            = (unsigned char *)malloc(N*sizeof(unsigned char ));
	ytemp            = (char *)malloc(N*sizeof(char ));
#endif

#ifdef OMP 
    num_threads      = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif


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


	for(i = 0 ; i < N ; i++)
	{		
		w[i]            = cteN;	
	}

	for(i = 0 ; i < d ; i++)
	{		
		indexF[i]       = i;
	}


	/* Previous premodel */

#ifdef OMP 
#pragma omp parallel for default(none) private(fm,j,i,ind,featuresIdx_opt,th_opt,a_opt,indM) shared(premodel,Npremodel,N,Xt,y,w) reduction (+:sumw) 
#endif
	for(j = 0 ; j < Npremodel ; j++)
	{
		indM             = j*4;
		featuresIdx_opt  = ((int) premodel[0 + indM]) - 1;	
		th_opt           = (char) premodel[1 + indM];
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
#pragma omp parallel default(none) private(xtemp,wtemp,ytemp,j,i,ind,indice,Errplus,Errminus,Splus,Sminus,Tplus,Tminus,indN) shared(d,N,N1,indexF,idX,Xt,w,y,featuresIdx_opt,th_opt,a_opt,Errormin)
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
				if(indexF[j] != -1)
				{
					for(i = 0 ; i < N ; i++)	
					{
						ind         = i + indN;
						indice      = (int)idX[ind];
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
							a_opt           = 1;	
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
							a_opt           = -1;	
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

		ind                       = featuresIdx_opt*N;
		errm                      = 0.0;

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

		indexF[featuresIdx_opt]   = -1;
		param[0 + indM]           = (double) (featuresIdx_opt + 1);
		param[1 + indM]           = (double) th_opt;
		param[2 + indM]           = a_opt*cm;
		param[3 + indM]           = 0.0;
		indM                      += 4;
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
	int i , j , jm = 0, in;

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

