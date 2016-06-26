
/*

  Predict data label with a Strong Classifier trained with haar_adaboost_binary_model_cascade

  Usage
  ------

  [yest , fx] = haar_adaboost_binary_predict_cascade(II , [options]);

  
  Inputs
  -------

  II                                    Integral of standardized Images (Ny x Nx x N) standardized in DOUBLE format

  options
          param                         Trained parameters matrix (4 x T). model(: , i) = [featureIdx ; th ; a ; b]
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)
			                            b                         Zeros (1 x T), i.e. b = zeros(1 , T)
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
          F                             Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle
          weaklearner                   Choice of the weak learner used in the training phase (default weaklearner = 2)
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
          cascade_type                  Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)
          cascade                       Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
									    If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))	
									    cascade(2 , :) reprensent thresholds for each segment

  Outputs
  -------
  
  yest                                  Estimated labels (1 x N) in INT8 format
  fx                                    Additive models (1 x N)


  To compile
  ----------

  mex  -output haar_adaboost_binary_predict_cascade.dll haar_adaboost_binary_predict_cascade.c

  mex  -f mexopts_intel10.bat -output haar_adaboost_binary_predict_cascade.dll haar_adaboost_binary_predict_cascade.c


  Example 1    Viola-Jones database
  ---------

  clear, close all
  load viola_24x24
  y                         = int8(y);
  options                   = load('haar_dico_2.mat');

  II                        = image_integral_standard(X);
  [Ny , Nx , P]             = size(II);

  options.T                 = 10;
  options.F                 = haar_featlist(Ny , Nx , options.rect_param);
  
  N                         = 1200;
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
  plot(fpp , tpp)
  axis([-0.02 , 1.02 , -0.02 , 1.02])


  [dum , ind]              = sort(ytest , 'descend');
  figure(2)
  plot(fxtest(ind))


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
	double        *rect_param;
	int            nR;
	unsigned int  *F;
	int            nF;
    int            weaklearner;
    int            cascade_type;
	double        *cascade;
	int            Ncascade;
};
/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
double Area(double * , int , int , int , int , int );
double haar_feat(double *  , int  , double * , unsigned int * , int , int , int );
void  haar_adaboost_binary_predict_cascade(double * , int , int , int , struct opts , char *, double *);
/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    double *II ;
	const int *dimsII;
	double  param_default[400]     = {10992.0000,-6.1626,-0.7788,0.0000,76371.0000,24.2334,0.6785,0.0000,4623.0000,-0.6328,-0.5489,0.0000,58198.0000,-4.4234,-0.5018,0.0000,67935.0000,-12.8916,-0.4099,0.0000,19360.0000,-1.7222,0.1743,0.0000,60243.0000,-1.9187,-0.2518,0.0000,3737.0000,-0.9260,0.1791,0.0000,58281.0000,-16.1455,-0.2447,0.0000,13245.0000,-1.6818,0.2183,0.0000,4459.0000,1.7972,0.2043,0.0000,7765.0000,3.0506,0.1665,0.0000,10105.0000,-2.0763,0.1764,0.0000,2301.0000,-2.4221,-0.1526,0.0000,4250.0000,-0.2044,0.1077,0.0000,59328.0000,24.8328,0.2129,0.0000,10127.0000,-2.1996,0.1746,0.0000,65144.0000,-35.6228,-0.2307,0.0000,43255.0000,-0.5288,0.1970,0.0000,57175.0000,-0.2119,0.0597,0.0000,59724.0000,-27.5468,-0.2059,0.0000,13278.0000,-2.1100,0.1895,0.0000,55098.0000,22.4124,0.1913,0.0000,13238.0000,-1.7093,0.1707,0.0000,62386.0000,0.3067,0.1283,0.0000,24039.0000,6.9595,0.1639,0.0000,43211.0000,-0.5982,0.1188,0.0000,62852.0000,9.6709,0.1652,0.0000,43236.0000,-0.6296,0.1530,0.0000,45833.0000,1.7152,0.1974,0.0000,7095.0000,-1.2430,0.1269,0.0000,76347.0000,-27.4002,-0.1801,0.0000,3737.0000,-0.8826,0.1462,0.0000,65143.0000,36.2253,0.1581,0.0000,13160.0000,-2.5302,0.1469,0.0000,4845.0000,0.7053,-0.0690,0.0000,52810.0000,-13.5220,-0.1594,0.0000,43234.0000,0.5907,-0.1420,0.0000,60847.0000,39.2252,0.1563,0.0000,43234.0000,-0.5423,0.1417,0.0000,56659.0000,0.7945,0.1387,0.0000,56930.0000,-1.3875,-0.1496,0.0000,13224.0000,-1.5798,0.1080,0.0000,63154.0000,14.8166,0.1961,0.0000,13162.0000,-2.3354,0.1639,0.0000,10722.0000,0.6559,0.2141,0.0000,7528.0000,1.1026,0.1077,0.0000,4263.0000,0.1324,-0.0485,0.0000,45151.0000,-1.4198,-0.1234,0.0000,7095.0000,1.5141,-0.1367,0.0000,68446.0000,-25.0890,-0.1744,0.0000,43277.0000,-0.5919,0.1564,0.0000,3613.0000,-0.5823,-0.1439,0.0000,5418.0000,3.9535,-0.1502,0.0000,58985.0000,24.7405,0.1754,0.0000,43785.0000,-0.9376,0.1194,0.0000,46582.0000,-5.8589,-0.1286,0.0000,43470.0000,-0.6392,0.1396,0.0000,10262.0000,-2.9209,-0.1251,0.0000,10105.0000,-2.0250,0.0960,0.0000,3555.0000,0.7341,0.1348,0.0000,10115.0000,1.6321,-0.1274,0.0000,76579.0000,-39.8316,-0.1442,0.0000,10228.0000,1.8771,-0.1245,0.0000,57005.0000,2.3937,0.1431,0.0000,43830.0000,-0.7996,0.0652,0.0000,48673.0000,8.8965,0.1181,0.0000,18845.0000,-2.2572,0.0872,0.0000,50225.0000,-1.5850,-0.1181,0.0000,43284.0000,0.5782,-0.1278,0.0000,72000.0000,-8.5961,-0.1282,0.0000,43214.0000,-0.6367,0.1053,0.0000,72559.0000,23.7860,0.1368,0.0000,43792.0000,1.0846,-0.1150,0.0000,56537.0000,-0.1965,-0.1262,0.0000,13421.0000,8.9499,0.0433,0.0000,172.0000,0.5319,-0.0946,0.0000,68220.0000,20.5078,0.1688,0.0000,16105.0000,-1.8842,0.1081,0.0000,79153.0000,5.8776,0.1301,0.0000,19180.0000,2.0606,0.1314,0.0000,13.0000,0.5438,-0.0802,0.0000,67201.0000,-6.6425,-0.1443,0.0000,43210.0000,0.5881,-0.1349,0.0000,65075.0000,-44.1279,-0.1170,0.0000,43214.0000,0.5392,-0.0840,0.0000,139.0000,0.2841,0.1480,0.0000,10209.0000,1.8835,-0.0957,0.0000,44409.0000,-1.0357,-0.1648,0.0000,43210.0000,-0.5252,0.1002,0.0000,47431.0000,6.8252,0.0927,0.0000,10235.0000,-1.3325,0.0795,0.0000,14896.0000,-12.2989,-0.0802,0.0000,1752.0000,-0.8487,-0.1193,0.0000,6964.0000,-1.7033,0.0944,0.0000,64124.0000,32.1583,0.1058,0.0000,43215.0000,-0.6988,0.0997,0.0000,76579.0000,-39.5722,-0.0932,0.0000,43966.0000,1.0216,-0.0926,0.0000,68446.0000,-25.5175,-0.1295,0.0000};
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	struct opts options;
	int  Tcascade = 0;	
	int *dimsyest;
    char *yest;
	double *fx;
	int i , Ny , Nx , N ;
	mxArray *mxtemp;
	double *tmp;
	int tempint;

	options.nR           = 4;
    options.nF           = 0;
	options.weaklearner  = 2; 
	options.cascade_type = 0;
	options.Ncascade     = 1;
	options.T            = 10;
	
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
			options.T                     = 10;
		}


		mxtemp                             = mxGetField( prhs[1] , 0, "rect_param" );
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

		mxtemp                             = mxGetField( prhs[1] , 0, "F" );
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
		options.T                     = 10;

		options.rect_param            = (double *)mxMalloc(40*sizeof(double));	
		for (i = 0 ; i < 40 ; i++)
		{		
			options.rect_param[i]     = rect_param_default[i];
		}	

		options.nF                    = number_haar_features(Ny , Nx , options.rect_param , options.nR);
		options.F                     = (unsigned int *)mxMalloc(5*options.nF*sizeof(unsigned int));
		haar_featlist(Ny , Nx , options.rect_param , options.nR , options.F);

		options.cascade               = (double *)mxMalloc(2*sizeof(double));
		options.cascade[0]            = (double) options.T;
		options.cascade[1]            = 0.0;		
	}
	
    /*----------------------- Outputs -------------------------------*/

	/* Output 1  */
	
	dimsyest                              = (int *)mxMalloc(2*sizeof(int));
	dimsyest[0]                           = 1;
	dimsyest[1]                           = N;
	plhs[0]                               = mxCreateNumericArray(2 , dimsyest , mxINT8_CLASS , mxREAL);
	yest                                  = (char *)mxGetPr(plhs[0]);
	
	/* Output 2  */
	
	plhs[1]                               =  mxCreateNumericMatrix(1 , N, mxDOUBLE_CLASS, mxREAL);
	fx                                    =  mxGetPr(plhs[1]);
	
	/*------------------------ Main Call ----------------------------*/
	
	haar_adaboost_binary_predict_cascade(II , Ny , Nx , N , options , yest , fx  );

   /*--------------------------- Free memory -----------------------*/

	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( mxGetField( prhs[1] , 0 , "param" ) == NULL )	
		{
			mxFree(options.param);
		}

		if ( mxGetField( prhs[1] , 0 , "rect_param" ) == NULL )	
		{
			mxFree(options.rect_param);
		}

		if ( mxGetField( prhs[1] , 0 , "F" ) == NULL )	
		{
			mxFree(options.F);
		}

		if ( mxGetField( prhs[1] , 0 , "cascade" ) == NULL )	
		{
			mxFree(options.cascade);
		}
	}
	else
	{
		mxFree(options.param);
		mxFree(options.rect_param);
		mxFree(options.F);
		mxFree(options.cascade);
	}

	mxFree(dimsyest);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  haar_adaboost_binary_predict_cascade(double *II , int Ny , int Nx , int N , struct opts options , char *yest, double *fx)
{
	double *param = options.param , *rect_param = options.rect_param, *cascade = options.cascade;
	unsigned int *F = options.F;
	int nR = options.nR , nF = options.nF , Ncascade = options.Ncascade;
	int t , n , c  , Tc;
	int indc , indm , featureIdx , NyNx = Ny*Nx , indNyNx;
	double z;
	double  th , a  , thresc;
	int weaklearner = options.weaklearner , cascade_type = options.cascade_type;
	double sum , sum_total ;
	
	if(weaklearner == 2) /* Decision Stump */
	{
		indNyNx   = 0;

		for(n = 0 ; n < N ; n++)
		{
			
			sum_total  = 0.0;	
			indc       = 0;
			indm       = 0;
			
			for (c = 0 ; c < Ncascade ; c++)
			{
				Tc     = (int) cascade[0 + indc];
				thresc = cascade[1 + indc];
				sum    = 0.0;
				
				for(t = 0 ; t < Tc ; t++)
				{
					
					featureIdx = ((int) param[0 + indm]) - 1;

					z          = haar_feat(II + indNyNx , featureIdx , rect_param , F , Ny , nR , nF);		
					th         = param[1 + indm];
					a          = param[2 + indm];
					sum       += a*sign(z - th);
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
			
			if(cascade_type  == 1)
			{
				fx[n]       = sum_total;	
				yest[n]     = sign(sum_total);
			}
			else if(cascade_type  == 0)
			{
				fx[n]       = sum;	
				yest[n]     = sign(sum);
			}
					
			indNyNx    += NyNx;			
		}	
	}	
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
		return (II[(y+h1) + (x+w1)*Ny] - II[y1 + (x+w1)*Ny] - II[(y+h1) + x1*Ny] + II[y1 + x1*Ny]);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
