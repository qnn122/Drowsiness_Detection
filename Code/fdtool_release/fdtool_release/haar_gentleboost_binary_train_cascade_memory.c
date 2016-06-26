
/*

  Train precomputed Haar features with gentleboosting classifier for binary problem

  Usage
  ------

  param   = haar_gentleboost_binary_train_cascade_memory(X , y , [options]);


  Inputs
  -------

  X                                     Features matrix (Haar features) (d x N) (or (N x d) if transpose = 1) in SINGLE format
  y                                     Binary labels (1 x N), y[i] = {-1 , 1} in INT8 format
  options
         T                              Number of weak learners (default T = 100)
         weaklearner                    Choice of the weak learner used in the training phase (default weaklearner = 0)
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R     
	     lambda                         Regularization parameter for the perceptron's weights'update		  
	     max_ite                        Maximum number of iteration
		 epsi                           Sigmoid parameter
         premodel                       Classifier's premodels parameter up to n-1 stage (4 x Npremodels)(default premodel = [] for stage n=1)
		 transpose                      Suppose X' as input (in order to speed up Boosting algorithm avoiding internal transposing, default tranpose = 0)

If compiled with the "OMP" compilation flag

	     num_threads                    Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  param                                 param output (4 x T) for current stage n of the classifier's premodel
        featureIdx                      Feature indexes of the T best weaklearners (1 x T)
	    th                              Optimal Threshold parameters (1 x T)
	    a                               Affine parameter(1 x T)
	    b                               Bias parameter (1 x T)

  To compile
  ----------


  mex -g  -output haar_gentleboost_binary_train_cascade_memory.dll haar_gentleboost_binary_train_cascade_memory.c

  mex  -output haar_gentleboost_binary_train_cascade_memory.dll haar_gentleboost_binary_train_cascade_memory.c

  mex  -f mexopts_intel10.bat -output haar_gentleboost_binary_train_cascade_memory.dll haar_gentleboost_binary_train_cascade_memory.c

  If OMP directive is added, OpenMP support for multicore computation

  mex  -v -DOMP -f mexopts_intel10.bat -output haar_gentleboost_binary_train_cascade_memory.dll haar_gentleboost_binary_train_cascade_memory.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


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
  nb_examples           = 500;

  options               = load('haar_dico_2.mat');
  options.transpose     = 1;
  options.usesingle     = 1;
  options.T             = 10;
  options.F             = haar_featlist(Ny , Nx , options.rect_param);

  pos                   = find(y == 1);
  neg                   = find(y == -1);

  I                     = (X(: , : , [pos(1:nb_examples) , neg(1:nb_examples)]));
  y                     = y([pos(1:nb_examples) , neg(1:nb_examples)]);

  H                     = haar(I , options );


%  figure(1)
%  imagesc(H)
%  title('Haar Features')
%  drawnow


  index                 = randperm(length(y)); %shuffle data to avoid numerical discrepancies with long sequence of same label

  if(options.transpose)
    tic,options.param   = haar_gentleboost_binary_train_cascade_memory(H(index , :) , y(index) , options);,toc
  else
    tic,options.param   = haar_gentleboost_binary_train_cascade_memory(H(: , index) , y(index) , options);,toc
  end
  [yest , fx]           = haar_gentleboost_binary_predict_cascade_memory(H , options);
  
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
  title(sprintf('Best %d Haar features with Gentleboosting' , nb_feats) , 'fontsize' , 13)
  colormap(gray)


  figure(5)
  plot(abs(options.param(3 , :)) , 'linewidth' , 2)
  grid on
  xlabel('Weaklearner m')
  ylabel('|a_m|')

  Example 2
  ---------

  load viola_24x24

  y             = int8(y);

  Ny            = 24;
  Nx            = 24;
  N             = 8;
  Nimage        = 110;
  nb_feats      = 3;
  
  F             = mblbp_featlist(Ny , Nx);
  mapping       = getmapping(N,'u2');
  map           = uint8(mapping.table);
  map           = uint8(0:255);
  H             = mblbp(X , F , map);

  index         = randperm(length(y));


  tic,model     = haar_gentleboost_binary_train_cascade_memory(H(: , index) , y(index) , T);,toc

  [yest , fx]   = mblbp_gentleboost_binary_predict_cascade(H , model);
  indp          = find(y == 1);
  indn          = find(y ==-1);

  tp            = sum(yest(indp) == y(indp))/length(indp)
  fp            = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf          = sum(yest == y)/length(y)

  [tpp , fpp]   = basicroc(y , fx);

  figure(1)
  plot(fpp , tpp)
  axis([-0.02 , 1.02 , -0.02 , 1.02])


  [dum , ind]        = sort(y , 'descend');
  figure(2)
  plot(fx(ind))

  figure(3)

  best_feats = double(F(: , model(1 , 1:nb_feats)));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats(2,i)-best_feats(4,i) + 0.5,  best_feats(3,i)-best_feats(5,i) + 0.5 ,  3*best_feats(4,i) ,  3*best_feats(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d MBLBP features with Gentleboosting' , nb_feats) , 'fontsize' , 13)
  colormap(gray)


  Example 3
  ---------

  load viola_24x24


  close all
  load viola_24x24
  load model_detector_mblbp_24x24_3.mat
  Ny         = 24;
  Nx         = 24;
  Nimage     = 110;
  nb_feats   = 3;
  best_feats = model.F(: , model.param(1 , 1:nb_feats));
  I          = X(: , : , Nimage);
  imagesc(I)
  hold on
  for i = 1:nb_feats
   h = rectangle('Position', [best_feats(2,i)-best_feats(4,i) + 1,  best_feats(3,i)-best_feats(5,i) + 1 ,  3*best_feats(4,i) ,  3*best_feats(5,i)]);
   set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
  end
  hold off
  title(sprintf('Best %d MBLBP features' , nb_feats) , 'fontsize' , 13)
  colormap(gray)


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Changelog :  - Add OpenMP support
 ----------   - Add transpose option


 Reference ""


*/


#include <time.h>
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

#define znew   (z = 36969*(z&65535) + (z>>16) )
#define wnew   (w = 18000*(w&65535) + (w>>16) )
#define MWC    ((znew<<16) + wnew )
#define SHR3   ( jsr ^= (jsr<<17), jsr ^= (jsr>>13), jsr ^= (jsr<<5) )

#define randint SHR3
#define rand() (0.5 + (signed)randint*2.328306e-10)
#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))


#ifdef __x86_64__
    typedef int UL;
#else
    typedef unsigned long UL;
#endif

static UL jsrseed = 31340134 , jsr;

struct opts
{
	int            T;
    int            weaklearner;
	double         epsi;
	double         lambda;
	int            max_ite;
    double        *premodel;
    int            Npremodel;
	int            transpose;

#ifdef OMP 
    int   num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void randini(void);
void qsindex( float * , int * , int , int  );
void transposeX(float *, float * , int , int);
void gentleboost_decision_stump(float *, char *, int , int ,  struct opts , double *);
void gentleboost_perceptron(float *, char *, int , int ,  struct opts , double *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{		
    float *X ;	
	char *y;
	double *param;	
	int d , N; 
	mxArray *mxtemp;
	struct opts options ;
	double *tmp;
	int tempint;
	
	options.T           = 100;
	options.Npremodel   = 0;
	options.weaklearner = 0;
	options.epsi        = 1.0;
	options.lambda      = 1e-3;
	options.max_ite     = 10;
	options.transpose   = 0;

#ifdef OMP 
    options.num_threads = -1;
#endif
	
    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) ==2) && (!mxIsEmpty(prhs[0])) && (mxIsSingle(prhs[0])) )
	{
		X                                = (float *)mxGetData(prhs[0]);	
		d                                = mxGetM(prhs[0]);
		N                                = mxGetN(prhs[0]);
	}
	else
	{
		mexErrMsgTxt("X must be (d x N) or (N x d) in SINGLE format");		
	}
	
	/* Input 2  */
	
	if ( (nrhs > 1) && (!mxIsEmpty(prhs[1])) && (mxIsInt8(prhs[1])) )	
	{		
		y                                = (char *)mxGetData(prhs[1]);	
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
				mexPrintf("weaklearner = {0,1,2}, force to 0");		
				options.weaklearner       = 0;
			}
			else
			{
				options.weaklearner       = tempint;	
			}			
		}

		mxtemp                            = mxGetField(prhs[2] , 0 , "epsi");     
		if(mxtemp != NULL)
		{            
			tmp                           = mxGetPr(mxtemp);   
			options.epsi                  = tmp[0];
		}

		mxtemp                            = mxGetField(prhs[2] , 0 , "lambda");		
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			options.lambda                = tmp[0];
		}

		mxtemp                            = mxGetField(prhs[2] , 0 , "max_ite");	
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if(tempint < 1)
			{
				mexPrintf("max_ite > 0, force to default value");		
				options.max_ite           = 10;
			}
			else	
			{
				options.max_ite           =  tempint;		
			}
		}

		mxtemp                           = mxGetField(prhs[2] , 0 , "premodel");
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 4)
			{	
				mexErrMsgTxt("premodel must be (4 x Npremodel) in double format");
			}
			options.premodel             =  mxGetPr(mxtemp);
			options.Npremodel            =  mxGetN(mxtemp);
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

	if(options.weaklearner == 0)
	{
		plhs[0]              = mxCreateNumericMatrix(4 , options.T , mxDOUBLE_CLASS,mxREAL);
		param                = mxGetPr(plhs[0]);

		if(options.transpose)
		{
			gentleboost_decision_stump(X , y  , N , d, options, param);
		}
		else
		{
			gentleboost_decision_stump(X , y  , d , N, options, param);
		}
	}
	if(options.weaklearner == 1)
	{
		plhs[0]              = mxCreateNumericMatrix(4 , options.T , mxDOUBLE_CLASS,mxREAL);	
		param                = mxGetPr(plhs[0]);

		randini();
		if(options.transpose)
		{
			gentleboost_perceptron(X , y  , N , d, options, param);
		}
		else
		{
			gentleboost_perceptron(X , y  , d , N, options, param);
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  gentleboost_decision_stump(float *X , char *y , int d , int N , struct opts options , double *param)								 								 
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
	double *w , *wtemp ;
	float *Xt, *xtemp;
	char *ytemp;
	int *idX;
	double atemp , btemp  , sumSw , Eyw , fm  , sumwyy , error , errormin, th_opt , a_opt , b_opt;
	double Syw, Sw , temp;
	int *indexF;

	Xt               = (float *)malloc(Nd*sizeof(float ));
    idX              = (int *)malloc(Nd*sizeof(int));
	w                = (double *)malloc(N*sizeof(double));
	indexF           = (int *)malloc(d*sizeof(int));

#ifdef OMP 

#else
	wtemp            = (double *)malloc(N*sizeof(double));
	xtemp            = (float *)malloc(N*sizeof(float ));
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
#pragma omp parallel for default(none) private(j,i,ind,featuresIdx_opt,th_opt,a_opt,b_opt,fm,indM) shared(premodel,Npremodel,N,Xt,y,w) reduction (+:sumSw) 
#endif
	
	for(j = 0 ; j < Npremodel ; j++)
	{
		indM             = j*4;
		featuresIdx_opt  = ((int) premodel[0 + indM]) - 1;	
		th_opt           = premodel[1 + indM];
		a_opt            = premodel[2 + indM];
		b_opt            = premodel[3 + indM];
		ind              = featuresIdx_opt*N;
		
		sumSw            = 0.0;
		
		for (i = 0 ; i < N ; i++)
		{
			fm           = a_opt*(Xt[i + ind] > th_opt) + b_opt;	
			w[i]        *= exp(-y[i]*fm);
			sumSw       += w[i];
		}
		
		sumSw            = 1.0/(sumSw + verytiny);
		
		for (i = 0 ; i < N ; i++)
		{	
			w[i]         *= sumSw;
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
		Eyw              = 0.0;
		sumwyy           = 0.0;
		
		for(i = 0 ; i < N ; i++)	
		{
			temp        = y[i]*w[i];
			Eyw        += temp;
			sumwyy     += y[i]*temp;			
		}
		
		errormin         = huge;

#ifdef OMP 
#pragma omp parallel  default(none) private(xtemp,ind,indice,ytemp,wtemp,j,i,atemp,error,Syw,Sw,indN,btemp) shared(d,N,N1,indexF,idX,Xt,w,y,Eyw,sumwyy,featuresIdx_opt,th_opt,a_opt,b_opt, errormin)
#endif
		{
#ifdef OMP 
			wtemp               = (double *)malloc(N*sizeof(double));
			xtemp               = (float *)malloc(N*sizeof(float ));
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
						wtemp[i]    = w[indice];
						ytemp[i]    = y[indice];
					}

					Sw              = 0.0;	
					Syw             = 0.0;

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
		sumSw                     = 0.0;
#ifdef OMP 
#pragma omp parallel for default(none) private(i,fm,indice) shared (a_opt,ind,th_opt,b_opt,idX,Xt,w,y,N) reduction (+:sumSw)
#endif				
		for (i = 0 ; i < N ; i++)
		{
			fm          = a_opt*(Xt[i + ind] > th_opt) + b_opt;
			indice      =  idX[i+ind];
			w[indice]  *= exp(-y[indice]*fm);
			sumSw      += w[indice];

		}
		
		sumSw            = 1.0/(sumSw + verytiny);

#ifdef OMP 
#pragma omp parallel for default(none) private(i) shared (w,N,sumSw)
#endif	
		for (i = 0 ; i < N ; i++)
		{
			w[i]                 *= sumSw;
		}
		
		indexF[featuresIdx_opt]   = -1;
		param[0 + indM]           = (double) (featuresIdx_opt + 1);
		param[1 + indM]           = th_opt;
		param[2 + indM]           = a_opt;
		param[3 + indM]           = b_opt;
		indM                     += 4;
	}
	
	free(Xt);
	free(idX);
	free(w);
	free(indexF);	

#ifdef OMP

#else
	free(ytemp);
	free(xtemp);
	free(wtemp);
#endif
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  gentleboost_perceptron(float *X , char *y , int d , int N , struct opts options , double *param)
{
	double *premodel = options.premodel;
	int T  = options.T, Npremodel = options.Npremodel , max_ite = options.max_ite , transpose = options.transpose;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif

	double epsi = options.epsi, lambda = options.lambda;
	double error , errormin , cteN =1.0/(double)N;

	int t , j , i , k;	
	int indM , featuresIdx_opt, Nd = N*d , indN , index;

	float *Xt;
	double *w;
	double atemp , btemp , xi  , temp , fx , tempyifx , sum , fm;
	double a_opt , b_opt;
	int *indexF;


	Xt               = (float *)malloc(Nd*sizeof(float ));
	w                = (double *)malloc(N*sizeof(double ));
	indexF           = (int *)malloc(d*sizeof(int));

#ifdef OMP 
    num_threads      = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif


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
		w[i]     = cteN;
	}

	for(i = 0 ; i < d ; i++)
	{		
		indexF[i] = i;
	}


#ifdef OMP 
#pragma omp parallel for default(none) private(j,i,index,featuresIdx_opt,a_opt,b_opt,fm,indM) shared(premodel,Npremodel,N,Xt,y,w) reduction (+:sum) 
#endif
	for(j = 0 ; j < Npremodel ; j++)
	{	
		indM             = j*4;
		featuresIdx_opt  = ((int) premodel[0 + indM]) - 1;	
		a_opt            = premodel[2 + indM];
		b_opt            = premodel[3 + indM];

		index            = featuresIdx_opt*N;
		sum              = 0.0;

		for (i = 0 ; i < N ; i++)
		{
			fm           = a_opt*Xt[i + index] + b_opt;	
			w[i]        *= exp(-y[i]*fm);
			sum         += w[i];
		}

		sum              = 1.0/(sum + verytiny);
		for (i = 0 ; i < N ; i++)
		{
			w[i]         *= sum;
		}	
	}

	indM    = 0;

	for(t = 0 ; t < T ; t++)		
	{
		errormin = huge;

#ifdef OMP 
#pragma omp parallel  for default(none) private(indN,j,i,k,index,xi,fx,temp,tempyifx) shared(d,N,indexF,Xt,w,y,featuresIdx_opt,a_opt,b_opt, errormin,jsr,epsi,lambda,max_ite) reduction(+:atemp,btemp,error)
#endif

		for(j = 0 ; j < d  ; j++)
		{
			indN         = j*N;
			if(indexF[j] != -1)
			{
				/* Random initialisation of weights */

				index  = floor(N*rand());	
				atemp  = Xt[index + indN];
				index  = floor(N*rand());
				btemp  = Xt[index + indN];

				/* Weight's optimization  */

				for(k = 0 ; k < max_ite ; k++)
				{	
					for(i = 0 ; i < N ; i++)		
					{

						xi         = Xt[i + indN];
						fx         = (2.0/( 1.0 + exp(-2.0*epsi*(atemp*xi + btemp)) )) - 1.0; /* sigmoid in [-1 , 1] */
						temp       = lambda*(y[i] - fx)*epsi*(1.0 - fx*fx);	/* d(sig(x))/dx = (1 - fx²) */				
						atemp     += (temp*xi);
						btemp     += temp;
					}
				}

				/* Weigthed error */

				error         = 0.0;

				for(i = 0 ; i < N ; i++)		
				{
					fx        = (2.0/(1.0 + exp(-2.0*epsi*(atemp*Xt[i + indN] + btemp)))) - 1.0;
					tempyifx  = (y[i] - fx);
					error    += w[i]*tempyifx*tempyifx;
				}

				if(error < errormin)		
				{
					errormin        = error;	
					featuresIdx_opt = j;
					a_opt           = atemp;
					b_opt           = btemp;
				}	
			}	
		}

		index                     = featuresIdx_opt*N;
		sum                       = 0.0;

#ifdef OMP 
#pragma omp parallel  for default(none) private(i,fm) shared(Xt,w,y,N,epsi,a_opt,b_opt,index) reduction(+:sum)
#endif
		for (i = 0 ; i < N ; i++)
		{
			fm           = (2.0/(1.0 + exp(-2.0*epsi*(a_opt*Xt[i + index] + b_opt)))) - 1.0;	
			w[i]        *= exp(-y[i]*fm);
			sum         += w[i];
		}

		sum              = 1.0/(sum + verytiny);

#ifdef OMP 
#pragma omp parallel  for default(none) private(i) shared(w,N,sum)
#endif
		for (i = 0 ; i < N ; i++)
		{	
			w[i]         *= sum;
		}

		indexF[featuresIdx_opt]   = -1;
		param[0 + indM]           = (double) (featuresIdx_opt + 1);
		param[1 + indM]           = 0.0;
		param[2 + indM]           = a_opt;
		param[3 + indM]           = b_opt;

		indM                     += 4;
	}
	
	free(Xt);
	free(w);	
	free(indexF);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void qsindex (float  *a, int *index , int lo, int hi)
{
	/*  lo is the lower index, hi is the upper index
	of the region of array a that is to be sorted

	*/
	int i=lo, j=hi , ind;
	float x=a[(lo+hi)/2] , h;

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
void transposeX(float *A, float *B , int m , int n)
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
void randini(void)
{
     /* SHR3 Seed initialization */
    
    jsrseed  = (UL) time( NULL );
    jsr     ^= jsrseed;
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
