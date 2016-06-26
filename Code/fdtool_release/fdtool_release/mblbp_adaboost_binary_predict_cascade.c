
/*

  Predict data label with a Strong Classifier trained with mblbp_gentleboost_binary_train_cascade

  Usage
  ------

  [yest , fx] = mblbp_adaboost_binary_predict_cascade(X , [options]);

  
  Inputs
  -------

  X                                    Features matrix (d x N) in INT8 format
  options
          param                        Trained param structure where param(: , i) = [featureIdx ; th ; a ; b]
                                       featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                           th                        Optimal Threshold parameters (1 x T)
			                           a                         Affine parameter(1 x T)
			                           b                         Bias parameter (1 x T)
          weaklearner                  Choice of the weak learner used in the training phase (default weaklearner = 2)
			                           weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
          cascade_type                 Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)
          cascade                      Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                       If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
									   If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))
									   cascade(2 , :) reprensent thresholds for each segment
		  transpose                    Suppose X' as input (in order to speed up Boosting algorithm avoiding internal transposing, default tranpose = 0)

  Outputs
  -------
  yest                                 Estimated labels (1 x N) in INT8 format
  fx                                   Additive models (1 x N)

  To compile
  ----------

  mex  -output mblbp_adaboost_binary_predict_cascade.dll mblbp_adaboost_binary_predict_cascade.c
  mex  -f mexopts_intel10.bat -output mblbp_adaboost_binary_predict_cascade.dll mblbp_adaboost_binary_predict_cascade.c

  Example 1
  ---------


 load viola_24x24.mat

 Ny                           = 24;
 Nx                           = 24;
 Nimage                       = (100:120);
 options.F                    = mblbp_featlist(Ny , Nx);
 H                            = mblbp(H , options);
 [yest , fx]                  = mblbp_adaboost_binary_predict_cascade(z , options);

 options.cascade              = [10 , 10 ; 0 , 0];

 [yest_cascade , fx_cascade]  = mblbp_adaboost_binary_predict_cascade(z , model.param , [] , cascade);


 disp([yest ; yest1])



 load test_mblbp.mat

 [yest , fx]             = mblbp_adaboost_binary_predict_cascade(H , model);



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
    int            cascade_type;
	double        *cascade;
	int            Ncascade;
	int            transpose;
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void  adaboost_binary_model_cascade(unsigned char * , int , int , struct opts , char * , double *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned char *X ;
	int Tcascade = 0;
	int *dimsyest;
	double  param_default[400]      = {4608.000000,240.000000, 0.698196, 0.000000,8396.000000, 6.000000,-0.632038, 0.000000,4717.000000,13.000000,-0.475997, 0.000000,4776.000000,128.000000,-0.417231, 0.000000,986.000000,240.000000, 0.371072, 0.000000,2431.000000,192.000000, 0.365243, 0.000000,6463.000000,62.000000,-0.300767, 0.000000,2399.000000,192.000000, 0.283725, 0.000000,4630.000000,40.000000,-0.272978, 0.000000,4215.000000,161.000000, 0.277569, 0.000000,3030.000000,120.000000, 0.280558, 0.000000,655.000000,28.000000,-0.273113, 0.000000,604.000000,240.000000, 0.257744, 0.000000,185.000000,32.000000, 0.244766, 0.000000,2274.000000,195.000000, 0.224438, 0.000000,5389.000000,221.000000, 0.234751, 0.000000,4100.000000,156.000000, 0.224505, 0.000000,95.000000,191.000000, 0.214761, 0.000000,570.000000,62.000000,-0.207912, 0.000000,303.000000,113.000000, 0.224903, 0.000000,7335.000000,88.000000,-0.225694, 0.000000,268.000000,128.000000, 0.212380, 0.000000,5580.000000,193.000000, 0.184738, 0.000000,7362.000000,62.000000,-0.215452, 0.000000,3290.000000,206.000000, 0.189538, 0.000000,3065.000000,225.000000,-0.187511, 0.000000,996.000000,31.000000,-0.202382, 0.000000,3175.000000,245.000000, 0.186089, 0.000000,446.000000,26.000000, 0.197077, 0.000000,5027.000000, 1.000000,-0.189798, 0.000000,1611.000000,56.000000, 0.185341, 0.000000,2553.000000,241.000000, 0.194732, 0.000000,2595.000000,143.000000,-0.179444, 0.000000,4151.000000,60.000000,-0.141295, 0.000000,74.000000,192.000000, 0.181271, 0.000000,350.000000,56.000000, 0.179354, 0.000000,5575.000000,63.000000,-0.188858, 0.000000,8417.000000,223.000000, 0.184274, 0.000000,2726.000000,215.000000,-0.205059, 0.000000,8090.000000, 3.000000,-0.188663, 0.000000,619.000000,159.000000,-0.184901, 0.000000,273.000000,221.000000, 0.177643, 0.000000,1299.000000,96.000000,-0.153172, 0.000000,7658.000000,193.000000, 0.182067, 0.000000,3161.000000,36.000000, 0.179439, 0.000000,5600.000000,14.000000,-0.192280, 0.000000,6205.000000,119.000000,-0.131479, 0.000000,2318.000000,223.000000, 0.156541, 0.000000,184.000000,31.000000, 0.173775, 0.000000,193.000000,240.000000, 0.178032, 0.000000,5782.000000,64.000000,-0.151024, 0.000000,3539.000000,226.000000,-0.187696, 0.000000,1139.000000,227.000000, 0.179310, 0.000000,2447.000000,156.000000, 0.170499, 0.000000,822.000000,212.000000,-0.178263, 0.000000,6274.000000,68.000000,-0.181376, 0.000000,4037.000000,193.000000, 0.167209, 0.000000,2422.000000,192.000000, 0.167217, 0.000000,1309.000000,192.000000, 0.168216, 0.000000,3801.000000,63.000000,-0.170634, 0.000000,67.000000,215.000000,-0.171848, 0.000000,5999.000000,248.000000, 0.172926, 0.000000,207.000000,32.000000, 0.158846, 0.000000,4362.000000, 4.000000,-0.172901, 0.000000,163.000000,56.000000, 0.180091, 0.000000,910.000000,223.000000, 0.179276, 0.000000,394.000000,219.000000,-0.157030, 0.000000,315.000000,240.000000, 0.167899, 0.000000,302.000000,111.000000, 0.168984, 0.000000,8144.000000,62.000000,-0.130941, 0.000000,352.000000,196.000000,-0.160475, 0.000000,6015.000000,248.000000, 0.163526, 0.000000,327.000000,56.000000, 0.159658, 0.000000,3343.000000,100.000000,-0.159380, 0.000000,2770.000000,28.000000, 0.155594, 0.000000,3418.000000, 8.000000,-0.171566, 0.000000,1031.000000,152.000000,-0.158116, 0.000000,1224.000000,191.000000, 0.173054, 0.000000,5586.000000,62.000000,-0.144778, 0.000000,4734.000000,163.000000, 0.163755, 0.000000,306.000000,60.000000, 0.158639, 0.000000,3168.000000,39.000000,-0.163906, 0.000000,4161.000000,193.000000, 0.155211, 0.000000,6779.000000,39.000000, 0.161238, 0.000000,2617.000000,107.000000,-0.162070, 0.000000,2218.000000,224.000000, 0.160180, 0.000000,774.000000,120.000000, 0.158661, 0.000000,3834.000000,60.000000,-0.147024, 0.000000,2130.000000,207.000000, 0.169375, 0.000000,3082.000000,226.000000,-0.167427, 0.000000,406.000000,192.000000, 0.158246, 0.000000,295.000000,205.000000, 0.153665, 0.000000,330.000000,201.000000,-0.159461, 0.000000,173.000000,224.000000, 0.169945, 0.000000,1157.000000,140.000000,-0.147730, 0.000000,5520.000000,32.000000,-0.146256, 0.000000,2092.000000,63.000000,-0.160068, 0.000000,792.000000,120.000000, 0.156527, 0.000000,117.000000,239.000000, 0.129437, 0.000000,2402.000000,26.000000, 0.154331, 0.000000};
	struct opts options;
    char *yest;
	double *fx;
	int i , d , N;
	mxArray *mxtemp;
	double *tmp;
	int tempint;
	
	options.weaklearner  = 2; 
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
				Tcascade         += (int) options.cascade[i];
			}

			if(Tcascade > options.T)
			{
				mexErrMsgTxt("sum(cascade(1 , :)) <= T");
			}
		}
		else
		{
			options.cascade                           = (double *)mxMalloc(2*sizeof(double));
			options.cascade[0]                        = (double) options.T;
			options.cascade[1]                        = 0.0;		
		}	
	}
	
	else
	{
		options.param                 = (double *)mxMalloc(400*sizeof(double));	
		for(i = 0 ; i < 400 ; i++)
		{
			options.param[i]          = param_default[i];	
		}	
		options.T                     = 100;

		options.cascade               = (double *)mxMalloc(2*sizeof(double));
		options.cascade[0]            = (double) options.T;
		options.cascade[1]            = 0.0;		
	}
	

	
	
  /*----------------------- Outputs -------------------------------*/	

	/* Output 1  */

	if(options.transpose)
	{

		dimsyest                              = (int *)mxMalloc(2*sizeof(int));	
		dimsyest[0]                           = 1;
		dimsyest[1]                           = d;
		plhs[0]                               = mxCreateNumericArray(2 , dimsyest , mxINT8_CLASS , mxREAL);
		yest                                  = (char *)mxGetPr(plhs[0]);

		/* Output 2  */

		plhs[1]                               =  mxCreateNumericMatrix(1 , d, mxDOUBLE_CLASS, mxREAL);
		fx                                    =  mxGetPr(plhs[1]);

		/*------------------------ Main Call ----------------------------*/

		adaboost_binary_model_cascade(X , N , d , options , yest , fx );
	}
	else
	{
		dimsyest                              = (int *)mxMalloc(2*sizeof(int));	
		dimsyest[0]                           = 1;
		dimsyest[1]                           = N;
		plhs[0]                               = mxCreateNumericArray(2 , dimsyest , mxINT8_CLASS , mxREAL);
		yest                                  = (char *)mxGetPr(plhs[0]);

		/* Output 2  */

		plhs[1]                               =  mxCreateNumericMatrix(1 , N, mxDOUBLE_CLASS, mxREAL);
		fx                                    =  mxGetPr(plhs[1]);

		/*------------------------ Main Call ----------------------------*/

		adaboost_binary_model_cascade(X , d , N , options , yest , fx );

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
void  adaboost_binary_model_cascade(unsigned char *X , int d , int N , struct opts options , char *yest, double *fx)									
{
	double *param = options.param , *cascade = options.cascade;
	int weaklearner = options.weaklearner , cascade_type = options.cascade_type , Ncascade = options.Ncascade , transpose = options.transpose;
	int t , n , c  , Tc;
	int indd = 0 , indc , indm , featureIdx;
	double  th , a  , thresc;		
	double sum , sum_total ;


	if(weaklearner == 2) /* Decision Stump */
	{		
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
						th         = (unsigned char) param[1 + indm];
						a          = param[2 + indm];
						sum       += a*sign(X[featureIdx*N + n] - th);
						indm      += 4;
					}

					sum_total     += sum;

					if((sum_total < thresc) && (cascade_type == 1))		
					{
						break;	
					}
					else if((sum < thresc) && (cascade_type == 0))
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
				else if(cascade_type == 0 )
				{
					fx[n]       = sum;	
					yest[n]     = sign(sum);
				}
			}	
		}
		else
		{
			indd      = 0;	
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
						th         = (unsigned char) param[1 + indm];
						a          = param[2 + indm];
						sum       += a*sign(X[featureIdx + indd] - th);
						indm      += 4;
					}

					sum_total     += sum;

					if((sum_total < thresc) && (cascade_type == 1))		
					{
						break;	
					}
					else if((sum < thresc) && (cascade_type == 0))
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
				else if(cascade_type == 0 )
				{
					fx[n]       = sum;	
					yest[n]     = sign(sum);
				}
				indd       += d;
			}	
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
