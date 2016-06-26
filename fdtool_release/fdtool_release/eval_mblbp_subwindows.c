
/*

  Eval mblbp feature with a trained model for image X 

  Usage
  ------

  [fx , y]       = eval_mblbp_subwindows(X , model);

  
  Inputs
  -------

  I                                     Input image (Ny x N) in UINT8 format
  
  model                                 Trained model structure
       weaklearner                      Choice of the weak learner used in the training phase (default weaklearner = 0)
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
       param                            Trainned classfier parameters matrix (4 x T). Each row corresponds to :
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity when weaklearner = 2)
			                            b                         Offset (1 x T) (when weaklearner = 2, b = 0)
	   dimsItraining                    Size of the trainnig images used in the mblbp'model, i.e. (ny x nx) (default ny = 24, nx = 24)
       F                                Feature's parameters (5 x nF) in UINT32 format
       map                              Mapping of the mblbp used in the CLPB computation (1 x 256) in UINT8 format
       cascade_type                     Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade
       cascade                          Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
										If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))						
										cascade(2 , :) reprensent thresholds for each segment

If compiled with the "OMP" compilation flag

	   num_threads                      Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)


  Outputs
  -------
  fx                                    Output matrix (1 x 1) of the last stage/Strong classifier (cascade_type = 0/1) 
  y                                     Ispassing cascade vector(1 x 1), y=1 if yes, -1 otherwise 


  To compile
  ----------


  mex  -output eval_mblbp_subwindows.dll eval_mblbp_subwindows.c

  mex  -f mexopts_intel10.bat -output eval_mblbp_subwindows.dll eval_mblbp_subwindows.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output eval_mblbp_subwindows.dll eval_mblbp_subwindows.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

  Example 1    Viola-Jones database
  ---------


  I                = rgb2gray(imread('C:\utilisateurs\SeBy\Matlab\fdtool\images\train\positives\128_04223_m_60.png'));
  load model_detector_mblbp_24x24_4.mat
  thresh           = 0;
  [fx ,yfx]        = eval_mblbp_subwindows(I , model);
  yest             = int8(sign(fx - thresh));


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 02/20/2009

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


#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif
#define round(f)   ((f>=0)?(int)(f + .5):(int)(f - .5))
#define sign(a)    ((a) >= (0) ? (1.0) : (-1.0))
 
struct model
{
	int             weaklearner;
	double          epsi;
	double         *param;
	int             T;
	double         *dimsItraining;
	int             ny;
	int             nx;
	unsigned int   *F;
	int             nF;
	unsigned char  *map;
	double        *cascade;
	int            Ncascade;
	int            cascade_type;
#ifdef OMP 
    int            num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */
int Round(double );
int number_mblbp_features(int , int );
void mblbp_featlist(int  , int , unsigned int *);
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
void eval_mblbp_subwindows(unsigned char * , int , int , struct model  , double * , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned char *I;
	struct model detector;
    const int *dimsI ;
    int numdimsI , Tcascade = 0;
    double *fx , *y;
	mxArray *mxtemp;
    int i , Ny , Nx , powN  = 256  , tempint;
	double *tmp;
	double param_default[400]       = {4608.0000,240.0000, 1.1750,-0.6868,8396.0000, 7.0000,-1.0774, 0.6278,4717.0000,12.0000,-0.9667, 0.5909,6130.0000,223.5000, 0.7644,-0.3870,4776.0000,127.0000,-0.7657, 0.2433,5027.0000, 1.5000,-0.8168, 0.6111,2399.0000,192.0000, 0.6640,-0.3146,986.0000,240.0000, 0.6726,-0.2658,5289.0000,63.0000,-0.5540, 0.2837,1002.0000,227.5000, 0.6053,-0.2320,2571.0000,161.0000, 0.5391,-0.2587,6774.0000,46.0000,-0.5479, 0.3235,5389.0000,220.0000, 0.5034,-0.1697,986.0000,48.0000, 0.7172,-0.6259,3067.0000,234.0000,-0.7112, 0.0879,822.0000,228.0000,-0.6506, 0.0896,2411.0000,190.0000, 0.4595,-0.2182,185.0000,32.0000, 0.5747,-0.4633,4239.0000,143.0000,-0.5122, 0.1384,1309.0000,183.0000, 0.4818,-0.2120,4131.0000,64.0000,-0.4564, 0.2447,1145.0000,119.0000, 0.4988,-0.3431,2274.0000,195.0000, 0.4599,-0.2104,2753.0000,237.0000,-0.6033, 0.1039,2805.0000,24.0000, 0.6233,-0.5368,1611.0000,56.0000, 0.5743,-0.4798,5769.0000,225.0000,-0.5952, 0.0857,715.0000,128.0000, 0.4319,-0.2546,7914.0000,233.0000,-0.6488, 0.0804,2896.0000,241.0000, 0.4893,-0.1312,6555.0000,225.0000,-0.7045, 0.0771,2450.0000,14.0000, 0.6013,-0.5252,96.0000, 2.0000, 0.7296,-0.6644,1223.0000, 8.0000, 0.6591,-0.5927,308.0000, 2.0000, 0.7811,-0.7270,136.0000,207.0000,-0.5026, 0.0877,349.0000,237.0000,-0.5503, 0.0791,2164.0000,31.0000, 0.5243,-0.4402,4100.0000,157.0000, 0.3848,-0.1920,4078.0000,224.0000,-0.5150, 0.1033,2330.0000,60.0000, 0.4320,-0.3157,768.0000,252.0000,-0.8387, 0.0613,751.0000, 8.0000, 0.5779,-0.5144,1000.0000, 6.0000, 0.7844,-0.7250,74.0000, 8.0000, 0.5917,-0.5229,892.0000,191.0000, 0.3999,-0.1474,7986.0000,64.0000,-0.3796, 0.2177,2667.0000,48.0000, 0.4651,-0.3600,6691.0000,191.0000, 0.4077,-0.1707,2153.0000,63.0000,-0.3979, 0.2135,7633.0000,56.0000, 0.4288,-0.3232,6274.0000,68.0000,-0.3929, 0.2427,3818.0000,63.0000,-0.3822, 0.1872,586.0000,239.0000, 0.4054,-0.1366,3901.0000, 8.0000, 0.5937,-0.5174,238.0000,229.0000, 0.4361,-0.1072,2825.0000,95.0000,-0.3792, 0.2139,2031.0000,193.0000, 0.4069,-0.1689,2577.0000,143.0000,-0.4686, 0.1123,326.0000,55.0000, 0.4383,-0.3149,4119.0000,24.0000, 0.5279,-0.4414,928.0000,246.0000,-0.5931, 0.0655,2411.0000,227.0000,-0.5046, 0.0793,2334.0000,10.0000, 0.5918,-0.5350,2362.0000,11.0000, 0.5816,-0.5155,424.0000,25.0000, 0.4660,-0.3772,3088.0000,211.5000,-0.4360, 0.0976,2408.0000,26.0000, 0.5043,-0.4183,316.0000,223.0000, 0.3862,-0.1356,263.0000, 6.0000, 0.5788,-0.5087,2157.0000,227.0000,-0.5186, 0.0681,6581.0000,12.0000, 0.5558,-0.4945,1197.0000,227.0000,-0.4703, 0.0799,402.0000, 0.0000, 0.7701,-0.7340,2814.0000,12.0000, 0.5069,-0.4388,5812.0000,246.0000, 0.3962,-0.1081,197.0000,31.0000,-0.3743, 0.2453,2135.0000,12.0000, 0.5345,-0.4676,608.0000,249.0000,-0.7812, 0.0486,74.0000,62.0000,-0.3577, 0.2071,74.0000,193.0000, 0.3861,-0.1436,6545.0000,253.0000, 0.5373,-0.0608,302.0000,60.0000, 0.4138,-0.2837,2785.0000,192.0000, 0.3753,-0.1568,3153.0000,247.0000,-0.8017, 0.0470,7534.0000,14.0000,-0.4093, 0.2770,619.0000, 6.0000, 0.6074,-0.5501,2690.0000,252.0000,-0.6929, 0.0480,6657.0000,227.0000,-0.5275, 0.0568,5959.0000, 0.0000,-0.5938, 0.5327,163.0000,56.0000, 0.4318,-0.3468,3418.0000, 4.0000,-0.5357, 0.4684,1336.0000,246.0000, 0.4439,-0.0912,3609.0000, 0.0000,-0.5008, 0.4336,3013.0000,126.0000, 0.3753,-0.2499,1959.0000,244.0000, 0.5115,-0.0858,2703.0000,251.0000, 0.5093,-0.0679,5731.0000,253.0000,-0.7091, 0.0533,157.0000,131.0000,-0.3844, 0.1128,136.0000,249.0000,-0.6834, 0.0491};

	detector.weaklearner  = 0; 
	detector.epsi         = 0.1;
	detector.cascade_type = 0;
	detector.Ncascade     = 1;
	detector.nx           = 24;
	detector.ny           = 24;


#ifdef OMP 
    detector.num_threads  = -1;
#endif

    if ((nrhs < 2))       
    {	
        mexErrMsgTxt("At least 2 inputs are requiered for detector");	
	}
	
    /* Input 1  */

    dimsI                = mxGetDimensions(prhs[0]);
    numdimsI             = mxGetNumberOfDimensions(prhs[0]);
    
    if( !mxIsUint8(prhs[0]) )
    {      
        mexErrMsgTxt("I must be (Ny x Nx) in UINT8 format");   
    }

    Ny                   = dimsI[0];  
    Nx                   = dimsI[1];
    
    I                    = (unsigned char *)mxGetData(prhs[0]); 
 
    /* Input 2  */
    
    if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )   
    {
		mxtemp                            = mxGetField( prhs[1] , 0, "weaklearner" );	
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			
			if((tempint < 0) || (tempint > 3))
			{
				mexPrintf("weaklearner = {0,1,2}, force to 0");		
				detector.weaklearner      = 0;		
			}
			else
			{		
				detector.weaklearner      = tempint;	
			}			
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "epsi" );	
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			if(tmp[0] < 0.0 )
			{
				mexPrintf("epsi must be > 0, force to 0.1");		
				detector.epsi             = 0.1;
			}
			else
			{		
				detector.epsi             = tmp[0];	
			}			
		}

		mxtemp                             = mxGetField( prhs[1], 0, "param" );	
		if(mxtemp != NULL)
		{	
			detector.param                 = mxGetPr(mxtemp);
			detector.T                     = mxGetN(mxtemp);
		}
		else
		{
			detector.param                 = (double *)mxMalloc(400*sizeof(double));
			for(i = 0 ; i < 400 ; i++)
			{
				detector.param[i]          = param_default[i];	
			}	
			detector.T                     = 10;
		}

		mxtemp                             = mxGetField( prhs[1] , 0, "dimsItraining" );	
		if(mxtemp != NULL)
		{
			detector.dimsItraining         =  mxGetPr(mxtemp);              	
			detector.ny                    = (int)detector.dimsItraining[0];
			detector.nx                    = (int)detector.dimsItraining[1];
			if ((Ny < detector.ny ) || (Nx < detector.nx ))       
			{
				mexErrMsgTxt("I must be at least nyxnx");			
			}	
		}
		
		mxtemp                             = mxGetField( prhs[1] , 0, "F" );	
		if(mxtemp != NULL)
		{		
			detector.F                     = (unsigned int *) mxGetData(mxtemp);	
			detector.nF                    = mxGetN(mxtemp);
		}
		else		
		{
			detector.nF                    = number_mblbp_features(detector.ny , detector.nx);	
			detector.F                     = (unsigned int *)mxMalloc(5*detector.nF*sizeof(unsigned int));
			mblbp_featlist(Ny , Nx , detector.F);
		}
						
		mxtemp                             = mxGetField( prhs[1] , 0, "map" );		
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != powN)
			{		
				mexErrMsgTxt("map must be (1 x 256) in UINT8 format");	
			}
			
			detector.map                   = (unsigned char *) mxGetData(mxtemp);		
		}
		else
		{	
			detector.map                   = (unsigned char *)mxMalloc(powN*sizeof(unsigned char));	
			for(i = 0 ; i < powN ; i++)
			{
				detector.map[i]            = (unsigned char) i;	
			}	
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade_type" );
		if(mxtemp != NULL)
		{	
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			
			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("cascade_type = {0,1}, force to 0");	
				detector.cascade_type     = 0;	
			}
			else
			{			
				detector.cascade_type     = tempint;	
			}			
		}	

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade" );
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 2)
			{
				mexErrMsgTxt("cascade must be (2 x Ncascade)");		
			}

			detector.cascade               = mxGetPr(mxtemp );
			detector.Ncascade              = mxGetN(mxtemp);
			for(i = 0 ; i < 2*detector.Ncascade ; i=i+2)
			{

				Tcascade         += (int) detector.cascade[i];

			}
			if(Tcascade > detector.T)
			{
				mexErrMsgTxt("sum(cascade(1 , :)) <= T");

			}
		}
		else
		{
			detector.cascade                = (double *)mxMalloc(2*sizeof(double));
			detector.cascade[0]             = (double) detector.T;
			detector.cascade[1]             = 0.0;
			detector.Ncascade               = 1;
		}
#ifdef OMP 
		mxtemp                            = mxGetField( prhs[1] , 0, "num_threads" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			
			if((tempint < -2))
			{								
				detector.num_threads      = -1;
			}
			else
			{
				detector.num_threads      = tempint;	
			}			
		}
#endif
    }
	else
	{	
		detector.param                 = (double *)mxMalloc(400*sizeof(double));
		for(i = 0 ; i < 400 ; i++)
		{		
			detector.param[i]          = param_default[i];	
		}	
		detector.T                     = 10;
		
		detector.nF                    = number_mblbp_features(detector.ny , detector.nx);			
		detector.F                     = (unsigned int *)mxMalloc(5*detector.nF*sizeof(int));
		mblbp_featlist(detector.ny , detector.nx , detector.F);

		detector.map                   = (unsigned char *)mxMalloc(powN*sizeof(unsigned char));
		for(i = 0 ; i < powN ; i++)
		{
			detector.map[i]            = (unsigned char) i;	
		}

		detector.cascade               = (double *)mxMalloc(2*sizeof(double));
		detector.cascade[0]            = (double) detector.T;
		detector.cascade[1]            = 0.0;
		detector.Ncascade              = 1;
	}
    
    /*------------------------ Output ----------------------------*/

    plhs[0]                    = mxCreateDoubleMatrix(1 , 1 , mxREAL);  
    fx                         = mxGetPr(plhs[0]);

    plhs[1]                    = mxCreateDoubleMatrix(1 , 1 , mxREAL);
    y                          = mxGetPr(plhs[1]);

    /*------------------------ Main Call ----------------------------*/
	
	eval_mblbp_subwindows(I , Ny , Ny , detector , fx , y);
	
	/*--------------------------- Free memory -----------------------*/
	
	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "param" )) == NULL )
		{
			mxFree(detector.param);
		}	
		if ( (mxGetField( prhs[1] , 0 , "F" )) == NULL )
		{
			mxFree(detector.F);
		}
		if ( (mxGetField( prhs[1] , 0 , "map" )) == NULL )
		{
			mxFree(detector.map);
		}
		if ( (mxGetField( prhs[1] , 0 , "cascade" )) == NULL)   
		{
			mxFree(detector.cascade);
		}
	}
	else
	{
		mxFree(detector.param);
		mxFree(detector.cascade);
		mxFree(detector.F);
		mxFree(detector.map);
	}   
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void eval_mblbp_subwindows(unsigned char *I , int Ny , int Nx , struct model detector, double *fx , double *y)				
{
	double   *param = detector.param , *cascade = detector.cascade;
	unsigned char *map = detector.map;
	unsigned int *II , *Itemp , *F = detector.F;
	int ny = detector.ny , nx = detector.nx;
	int Ncascade = detector.Ncascade , weaklearner = detector.weaklearner , cascade_type = detector.cascade_type;
#ifdef OMP 
	int num_threads = detector.num_threads;
#endif
	double epsi = detector.epsi;
	double scalex = (Nx )/(double)nx , scaley = (Ny )/(double)ny;
	int xc , yc , xnw , ynw , xse , yse , w , h;
	unsigned int Ac;
	unsigned char valF , z;
	double sum , sum_total , a , b , th , thresc;
	int c , f , Tc , NyNx = Ny*Nx , indf , indc  , idxF;

#ifdef OMP 
	num_threads          = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
	omp_set_num_threads(num_threads);
#endif

	II            = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Itemp         = (unsigned int *) malloc(NyNx*sizeof(unsigned int));

	MakeIntegralImage(I  , II , Nx , Ny  , Itemp);	
	indf          = 0;
	sum_total     = 0.0;	
	y[0]          = 1.0;

	for (c = 0 ; c < Ncascade ; c++)
	{
		indc      = c*2;
		Tc        = (int) cascade[0 + indc];
		thresc    = cascade[1 + indc];
		sum       = 0.0;

#ifdef OMP 
#pragma omp parallel for default(none) private(f,idxF,th,a,b,xc,yc,w,h,xnw,ynw,xse,yse,Ac,valF,z) shared(II,Ncascade,Tc,param,F,Ny,scalex,scaley,map,weaklearner,epsi,cascade_type,sum_total,indf,sum) /* reduction (+:) */
#endif
		for (f = 0 ; f < Tc ; f++)
		{
			idxF  = ((int) param[0 + indf] - 1)*5;
			th    = param[1 + indf];
			a     = param[2 + indf];
			b     = param[3 + indf];

			xc    = Round(scalex*(F[1 + idxF]));
			yc    = Round(scaley*(F[2 + idxF]));
			w     = Round(scalex*F[3 + idxF]);
			h     = Round(scaley*F[4 + idxF]);

			xnw   = xc - w;
			ynw   = yc - h;
			xse   = xc + w;
			yse   = yc + h;

			Ac    = Area(II , xc  , yc  , w , h , Ny);

			valF  = 0;
			if(Area(II , xnw , ynw , w , h , Ny) > Ac)
			{
				valF |= 0x01;
			}
			if(Area(II , xc  , ynw , w , h , Ny)  > Ac)
			{
				valF |= 0x02;
			}
			if(Area(II , xse , ynw , w , h , Ny) > Ac)
			{
				valF |= 0x04;
			}
			if(Area(II , xse , yc  , w , h , Ny)  > Ac)
			{
				valF |= 0x08;
			}				
			if(Area(II , xse , yse , w , h , Ny) > Ac)
			{
				valF |= 0x10;
			}
			if(Area(II , xc  , yse , w , h , Ny) > Ac)
			{
				valF |= 0x020;
			}
			if(Area(II , xnw , yse , w , h , Ny) > Ac)
			{
				valF |= 0x040;
			}
			if(Area(II , xnw , yc  , w , h , Ny)  > Ac)
			{
				valF |= 0x080;
			}

			z           = map[valF];
			if(weaklearner == 0)
			{					
				sum    += (a*( z > th ) + b);	
			}
			if(weaklearner == 1)
			{
				sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a*z + b)))) - 1.0);
			}
			if(weaklearner == 2)
			{
				sum    += a*sign(z - th);
			}
			indf      += 4;
		}

		sum_total     += sum;

		if((sum_total < thresc) && (cascade_type == 1))	
		{
			y[0]      = -1.0;
			break;	
		}
		else if((sum < thresc) && (cascade_type == 0))
		{
			y[0]      = -1.0;
			break;	
		}
	}

	if(cascade_type == 1)
	{
		fx[0]     = sum_total;
	}
	else if(cascade_type == 0)
	{
		fx[0]     = sum;
	}

	free(II);
	free(Itemp);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void mblbp_featlist(int ny , int nx , unsigned int *F)
{
	int i , j , w = 1 , h , nofeat = 1 , co = 0; 	
	while(nx >= 3*w)
	{		
		h    = 1;		
		while(ny >= 3*h)
		{
			for (j = w ; j <= nx-2*w ; j++)
			{
				for (i = h ; i <= ny-2*h ; i++)
				{	
					F[0 + co] = nofeat;	
					F[1 + co] = j;
					F[2 + co] = i;
					F[3 + co] = w;
					F[4 + co] = h;
					co       += 5;
				}			
			}
			h++;
			nofeat++;		
		}
		w++;	
	}	
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int number_mblbp_features(int ny , int nx)
{
	int nF = 0 , X , Y  , nx1 = nx + 1 , ny1 = ny + 1 ;

	X           = (int) floor(nx/3);	
	Y           = (int) floor(ny/3);
	nF          = (int) (X*Y*(nx1 - (X+1)*1.5)*(ny1 - (Y+1)*1.5));
	return nF;
}
/*----------------------------------------------------------------------------------------------------------------------------------------------*/
void MakeIntegralImage(unsigned char *pIn, unsigned int *pOut, int iXmax, int iYmax , unsigned int *pTemp)
{
	/* Variable declaration */
	int x , y , indx = 0;
		
	for(x=0 ; x<iXmax ; x++)
	{
		pTemp[indx]     = (unsigned int)pIn[indx];
		indx           += iYmax;
	}
	for(y = 1 ; y<iYmax ; y++)
	{
		pTemp[y]        = pTemp[y - 1] + (unsigned int)pIn[y];
	}
	pOut[0]             = (unsigned int)pIn[0];
	indx                = iYmax;

	for(x=1 ; x<iXmax ; x++)
	{
		pOut[indx]      = pOut[indx - iYmax] + pTemp[indx];
		indx           += iYmax;
	}
	for(y = 1 ; y<iYmax ; y++)
	{
		pOut[y]         = pOut[y - 1] + (unsigned int)pIn[y];
	}
	
	/* Calculate integral image */

	indx                = iYmax;
	for(x = 1 ; x < iXmax ; x++)
	{
		for(y = 1 ; y < iYmax ; y++)
		{
			pTemp[y + indx]    = pTemp[y - 1 + indx] + (unsigned int)pIn[y + indx];	
			pOut[y + indx]     = pOut[y + indx - iYmax] + pTemp[y + indx];
		}
		indx += iYmax;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------------*/
unsigned int Area(unsigned int *II , int x , int y , int w , int h , int Ny)
{
	int h1 = h-1, w1 = w-1 , x1 = x-1, y1 = y-1;
	
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
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
int Round(double x)
{
	return ((int)(x + 0.5));
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
