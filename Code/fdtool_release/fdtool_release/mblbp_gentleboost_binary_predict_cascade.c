
/*

  Predict data label with a Strong Classifier trained with mblbp_gentleboost_binary_train_cascade

  Usage
  ------

  [yest , fx] = mblbp_gentleboost_binary_predict_cascade(X , [options]);
  
  Inputs
  -------

  X                                    Features matrix (d x N) (or (N x d) if transpose = 1) in INT8 format
  options
           param                       Trained param structure where param(: , i) = [featureIdx ; th ; a ; b]
                                       featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                           th                        Optimal Threshold parameters (1 x T)
			                           a                         Affine parameter(1 x T)
			                           b                         Bias parameter (1 x T)
           weaklearner                 Choice of the weak learner used in the training phase (default weaklearner = 0)
			                           weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                           weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(th,a)) = sigmoid(x ; a,b) in R
		   epsi                        Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1)
           cascade_type                Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)
           cascade                     Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                       If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
									   If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))					
									   cascade(2 , :) reprensent thresholds for each segment
		   transpose                   Suppose X' as input (in order to speed up Boosting algorithm avoiding internal transposing, default tranpose = 0)

  Outputs
  -------
  yest                                 Estimated labels (1 x N) in INT8 format
  fx                                   Additive models (1 x N)

  To compile
  ----------

  mex  -g -output mblbp_gentleboost_binary_predict_cascade.dll mblbp_gentleboost_binary_predict_cascade.c

  mex  -output mblbp_gentleboost_binary_predict_cascade.dll mblbp_gentleboost_binary_predict_cascade.c

  mex  -f mexopts_intel10.bat -output mblbp_gentleboost_binary_predict_cascade.dll mblbp_gentleboost_binary_predict_cascade.c

  mex  -v -DOMP -f mexopts_intel10.bat -output mblbp_gentleboost_binary_predict_cascade.dll mblbp_gentleboost_binary_predict_cascade.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


  Example 1
  ---------

  load wbc
  y(y==0)                 = -1;
  X                       = uint8(X);
  y                       = int8(y);

  otpions.T               = 8;
  options.weaklearner     = 0;
  options.cascade         = [3 , 5 ;0 , 0];
  param                   = mblbp_gentleboost_binary_train_cascade(X , y , options );
  [yest , fx]             = mblbp_gentleboost_binary_predict_cascade(X , options);
  [yest1 , fx1]           = mblbp_gentleboost_binary_predict_cascade(X , options);

  sum(yest == y)/length(y)
  sum(yest1 == y)/length(y)


  Example 2
  ---------

 load viola_24x24.mat
 load model_detector_mblbp_24x24_3.mat
 options.cascade         = [10 , 10 ; 0 , 0];

 Ny                      = 24;
 Nx                      = 24;
 Nimage                  = (100:120);
 F                       = mblbp_featlist(Ny , Nx);
 I                       = X(: , : , Nimage);
 z                       = mblbp(I , F);
 [yest , fx]             = mblbp_gentleboost_binary_predict_cascade(z , model.param , [] , cascade);
 [yest1 , fx1]           = mblbp_gentleboost_binary_predict_cascade(z , model.param);

 disp([yest ; yest1])



 load test_mblbp.mat

 [yest , fx]             = mblbp_gentleboost_binary_predict_cascade(H , options);


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009


 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999

              [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
			      ICB07

			  [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
			  [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004

*/

#include <math.h>
#include <mex.h>

#define sign(a) ((a) >= (0) ? (1) : (-1))

struct opts
{
    double        *param;
    int            T;
    int            weaklearner;
	double         epsi;
    int            cascade_type;
	double        *cascade;
	int            Ncascade;
	int            transpose;
};


/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void  gentleboost_binary_model_cascade(unsigned char * , int , int , struct opts , char * , double *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned char *X ;
	int Tcascade = 0;
	int *dimsyest;
	struct opts options;
	double  param_default[400]      = {4608.000000,240.000000, 1.173526,-0.696875,8396.000000, 6.000000,-1.078598, 0.638388,4717.000000,14.000000,-0.963634, 0.592582,6130.000000,223.000000, 0.772097,-0.397222,4776.000000,127.000000,-0.766392, 0.244775,5027.000000, 2.000000,-0.817259, 0.610746,2871.000000,230.000000, 0.674887,-0.265060,2399.000000,193.000000, 0.646071,-0.315706,6463.000000,63.000000,-0.603862, 0.265121,3405.000000,35.000000, 0.766934,-0.662389,4100.000000,156.000000, 0.540168,-0.301880,2571.000000,161.000000, 0.530798,-0.263511,446.000000,26.000000, 0.633320,-0.512410,4399.000000,227.000000,-0.676849, 0.091501,2594.000000,31.000000,-0.476431, 0.289892,3148.000000,20.000000, 0.804280,-0.730214,1145.000000,119.000000, 0.486766,-0.342477,2370.000000,197.000000,-0.544520, 0.130671,2274.000000,195.000000, 0.446675,-0.214572,185.000000,31.000000, 0.571244,-0.459783,5731.000000,103.000000,-0.447022, 0.209866,2385.000000,234.000000,-0.630079, 0.092474,637.000000,143.000000,-0.494488, 0.111781,394.000000,227.000000,-0.582216, 0.086135,163.000000,56.000000, 0.495774,-0.390787,746.000000,39.000000, 0.512376,-0.401526,2805.000000,22.000000, 0.595970,-0.514300,2082.000000,238.000000,-0.635780, 0.075970,619.000000,199.000000,-0.566791, 0.081756,4433.000000,62.000000,-0.420193, 0.249089,2060.000000,193.000000, 0.486046,-0.191050,429.000000,28.000000, 0.515364,-0.394730,566.000000,231.000000, 0.446127,-0.152200,207.000000,24.000000, 0.575127,-0.488578,7292.000000,235.000000,-0.613834, 0.070751,2034.000000,16.000000, 0.523964,-0.441340,2552.000000,227.000000, 0.435855,-0.133532,751.000000, 7.000000, 0.601553,-0.529733,639.000000,144.000000,-0.451953, 0.102912,1691.000000,231.000000, 0.406684,-0.166813,5769.000000,224.000000,-0.534844, 0.082547,715.000000,128.000000, 0.405626,-0.228827,7390.000000, 6.000000, 0.706080,-0.639602,3824.000000,66.000000,-0.420185, 0.249545,586.000000,239.000000, 0.436540,-0.154046,2063.000000,227.000000,-0.541510, 0.090198,2777.000000,28.000000, 0.481286,-0.383394,2422.000000,191.000000, 0.397001,-0.186291,3539.000000,236.000000,-0.525060, 0.090769,3933.000000, 0.000000, 0.756174,-0.708586,6647.000000,223.000000, 0.409855,-0.140120,7450.000000,228.000000,-0.602688, 0.076625,393.000000,19.000000, 0.502979,-0.421784,6274.000000,70.000000,-0.390106, 0.231661,306.000000,60.000000, 0.411627,-0.286630,6504.000000,34.000000,-0.389462, 0.254693,5820.000000,159.000000, 0.400407,-0.175723,4742.000000, 4.000000,-0.546041, 0.463564,6117.000000,32.000000, 0.658876,-0.597666,410.000000,227.000000,-0.584964, 0.072311,350.000000,56.000000, 0.431138,-0.310367,2057.000000,63.000000,-0.385852, 0.194165,3785.000000,193.000000, 0.417538,-0.159036,657.000000,175.000000,-0.458265, 0.102224,2059.000000,31.000000,-0.403317, 0.266542,175.000000, 6.000000, 0.552785,-0.480132,425.000000,14.000000, 0.547356,-0.475904,328.000000,225.000000,-0.522471, 0.089373,985.000000,246.000000, 0.413695,-0.119132,547.000000,24.000000, 0.537415,-0.457742,4472.000000,47.000000,-0.392839, 0.253257,2773.000000,231.000000,-0.525195, 0.085790,2471.000000,188.000000, 0.395191,-0.141898,2334.000000,14.000000, 0.547193,-0.471735,4739.000000,254.000000, 0.500081,-0.084969,2507.000000,62.000000,-0.370016, 0.212170,1112.000000,14.000000, 0.640565,-0.578802,785.000000,135.000000,-0.390474, 0.139785,784.000000,120.000000, 0.377614,-0.219883,1533.000000, 9.000000, 0.518969,-0.441034,2309.000000,243.000000,-0.575760, 0.064580,4015.000000,252.000000,-0.661391, 0.043609,2268.000000, 3.000000, 0.624063,-0.576244,2632.000000,252.000000,-0.743856, 0.047430,2374.000000,229.000000,-0.497562, 0.064330,334.000000, 6.000000, 0.570251,-0.512754,3303.000000,14.000000, 0.731640,-0.681881,2083.000000,33.000000,-0.382027, 0.241210,5281.000000,192.000000, 0.395277,-0.158413,5389.000000,157.000000, 0.374843,-0.184210,6215.000000,119.000000,-0.419879, 0.169990,155.000000,253.000000,-0.774667, 0.043396,1131.000000,240.000000, 0.396332,-0.135042,135.000000, 7.000000, 0.635912,-0.569856,136.000000,208.000000,-0.494439, 0.095380,2331.000000,36.000000, 0.421003,-0.317612,352.000000,196.000000,-0.430541, 0.100427,406.000000,192.000000, 0.378329,-0.144818,139.000000,120.000000, 0.377768,-0.229694,2130.000000,33.000000,-0.407584, 0.270431};
    char *yest;
	double *fx;
	int i , d , N;
	mxArray *mxtemp;
	double *tmp;
	int tempint;
	
	options.weaklearner  = 0; 
	options.cascade_type = 0;
	options.Ncascade     = 1;
	options.T            = 100;
	options.transpose    = 0;
	
    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) !=2) || !mxIsUint8(prhs[0]) )
	{	
		mexErrMsgTxt("X must be (d x N) in UINT8 format");	
	}
	X           = (unsigned char *)mxGetData(prhs[0]);
	d           = mxGetM(prhs[0]);
	N           = mxGetN(prhs[0]);
	
	/* Input 2  */
	
	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		mxtemp                             = mxGetField( prhs[1] , 0, "param" );
		if(mxtemp != NULL)
		{
			if (mxGetM(mxtemp) != 4)
			{	
				mexErrMsgTxt("model must be (4 x T) matrix");	
			}
			options.param                     = mxGetPr(mxtemp);
			options.T                         = mxGetN(mxtemp);
		}
		else
		{
			options.param                 = (double *)mxMalloc(400*sizeof(double));	
			for(i = 0 ; i < 400 ; i++)
			{
				options.param[i]          = param_default[i];	
			}	
			options.T                     = 100;
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "weaklearner" );
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

		mxtemp                            = mxGetField(prhs[1] , 0 , "epsi");
		if(mxtemp != NULL)
		{	
			tmp                           = mxGetPr(mxtemp);	
			options.epsi                  = tmp[0];	
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade_type" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			
			if((tempint < 0) || (tempint > 1))
			{
				options.cascade_type      = 0;	
			}
			else
			{
				options.cascade_type      = tempint;	
			}			
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "transpose" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];

			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("transpose = {0,1}, force to 0");		
				options.transpose         = 0;
			}
			else
			{
				options.transpose         = tempint;	
			}			
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade" );
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 2)
			{
				mexErrMsgTxt("cascade must be (2 x Ncascade)");		
			}

			options.cascade               = mxGetPr(mxtemp );
			options.Ncascade              = mxGetN(mxtemp);
			for(i = 0 ; i < 2*options.Ncascade ; i=i+2)
			{
				Tcascade                 += (int) options.cascade[i];
			}

			if(Tcascade > options.T)
			{
				mexErrMsgTxt("sum(cascade(1 , :)) <= T");
			}
		}
		else
		{
			options.cascade               = (double *)mxMalloc(2*sizeof(double));
			options.cascade[0]            = (double) options.T;
			options.cascade[1]            = 0.0;		
		}	
	}
	else
	{
		options.param                     = (double *)mxMalloc(400*sizeof(double));	
		for(i = 0 ; i < 400 ; i++)
		{
			options.param[i]              = param_default[i];	
		}	
		options.T                         = 100;

		options.cascade                   = (double *)mxMalloc(2*sizeof(double));
		options.cascade[0]                = (double) options.T;
		options.cascade[1]                = 0.0;		
	}
	
  /*----------------------- Outputs -------------------------------*/	
  if(options.transpose)
  {

	  /* Output 1  */

	  dimsyest                              = (int *)mxMalloc(2*sizeof(int));
	  dimsyest[0]                           = 1;
	  dimsyest[1]                           = d;
	  plhs[0]                               = mxCreateNumericArray(2 , dimsyest , mxINT8_CLASS , mxREAL);
	  yest                                  = (char *)mxGetPr(plhs[0]);

	  /* Output 2  */

	  plhs[1]                               =  mxCreateNumericMatrix(1 , d , mxDOUBLE_CLASS, mxREAL);
	  fx                                    =  mxGetPr(plhs[1]);

	  /*------------------------ Main Call ----------------------------*/

	  gentleboost_binary_model_cascade(X , N , d , options , yest , fx );
  }
  else
  {
	  dimsyest                              = (int *)mxMalloc(2*sizeof(int));
	  dimsyest[0]                           = 1;
	  dimsyest[1]                           = N;
	  plhs[0]                               = mxCreateNumericArray(2 , dimsyest , mxINT8_CLASS , mxREAL);
	  yest                                  = (char *)mxGetPr(plhs[0]);

	  /* Output 2  */

	  plhs[1]                               =  mxCreateNumericMatrix(1 , N , mxDOUBLE_CLASS, mxREAL);
	  fx                                    =  mxGetPr(plhs[1]);
	  /*------------------------ Main Call ----------------------------*/

	  gentleboost_binary_model_cascade(X , d , N , options , yest , fx );
  }
	
   /*--------------------------- Free memory -----------------------*/
	
	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( mxGetField( prhs[1] , 0 , "param" ) == NULL )	
		{
			mxFree(options.param);
		}

		if ( mxGetField( prhs[1] , 0 , "cascade" ) == NULL )	
		{
			mxFree(options.cascade);
		}
	}
	else
	{
		mxFree(options.param);
		mxFree(options.cascade);
	}	
	mxFree(dimsyest);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  gentleboost_binary_model_cascade(unsigned char *X , int d , int N , struct opts options , char *yest, double *fx)
{
	double *param = options.param , *cascade = options.cascade;
	int weaklearner = options.weaklearner , cascade_type = options.cascade_type , Ncascade = options.Ncascade , transpose = options.transpose;
	int t , n , c  , Tc;	
	int indd = 0, indc , indm , featureIdx;
	double  th , a , b , thresc;
	double epsi = options.epsi;
	double sum , sum_total ;

	if(transpose)
	{
		for(n = 0 ; n < N ; n++)
		{
			sum_total   = 0.0;
			indc        = 0;	
			indm        = 0;

			for (c = 0 ; c < Ncascade ; c++)
			{
				Tc     = (int) cascade[0 + indc];	
				thresc = cascade[1 + indc];
				sum    = 0.0;
				for(t = 0 ; t < Tc ; t++)
				{
					featureIdx = ((int) param[0 + indm]) - 1;	
					th         = param[1 + indm];
					a          = param[2 + indm];
					b          = param[3 + indm];

					if(weaklearner == 0) /* Decision Stump */
					{			
						sum   += (a*( X[featureIdx*N + n]>th ) + b);	
					}
					else if(weaklearner == 1) /* Perceptron */	
					{
						sum   += ((2.0/(1.0 + exp(-2.0*epsi*(a*X[featureIdx*N  + n] + b)))) - 1.0);	
					}
					indm      += 4;
				}

				sum_total     += sum;
				if((sum_total < thresc) && (cascade_type == 1))	
				{	
					break;	
				}
				else if( (sum < thresc) && (cascade_type == 0) )
				{	
					break;	
				}

				indc  += 2;
			}
			if(cascade_type == 1 )	
			{
				fx[n]       = sum_total;	
				yest[n]     = sign(sum_total);
			}
			else if(cascade_type == 0)
			{
				fx[n]       = sum;	
				yest[n]     = sign(sum);
			}
		}
	}
	else
	{
		for(n = 0 ; n < N ; n++)
		{
			sum_total   = 0.0;
			indc        = 0;	
			indm        = 0;

			for (c = 0 ; c < Ncascade ; c++)
			{
				Tc     = (int) cascade[0 + indc];	
				thresc = cascade[1 + indc];
				sum    = 0.0;
				for(t = 0 ; t < Tc ; t++)
				{
					featureIdx = ((int) param[0 + indm]) - 1;	
					th         = param[1 + indm];
					a          = param[2 + indm];
					b          = param[3 + indm];

					if(weaklearner == 0) /* Decision Stump */
					{			
						sum   += (a*( X[featureIdx + indd]>th ) + b);	
					}
					else if(weaklearner == 1) /* Perceptron */	
					{
						sum   += ((2.0/(1.0 + exp(-2.0*epsi*(a*X[featureIdx  + indd] + b)))) - 1.0);	
					}
					indm      += 4;
				}

				sum_total     += sum;
				if((sum_total < thresc) && (cascade_type == 1))	
				{	
					break;	
				}
				else if( (sum < thresc) && (cascade_type == 0) )
				{	
					break;	
				}

				indc  += 2;
			}
			if(cascade_type == 1 )	
			{
				fx[n]       = sum_total;	
				yest[n]     = sign(sum_total);
			}
			else if(cascade_type == 0)
			{
				fx[n]       = sum;	
				yest[n]     = sign(sum);
			}
			indd       += d;
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
