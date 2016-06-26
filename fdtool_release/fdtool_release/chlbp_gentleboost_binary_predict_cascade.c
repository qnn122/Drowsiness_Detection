/*

  Predict data label with a Strong Classifier trained with chlbp_gentleboost_binary_predict_cascade

  Usage
  ------

  [yest , fx] = chlbp_gentleboost_binary_predict_cascade(X , [options]);

  
  Inputs
  -------

  X                                    Features matrix (d x N) in INT32 format
  options
           param                       Trained param structure (4 x T)
           weaklearner                 Choice of the weak learner used in the training phase (default weaklearner = 0)
			                           weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                           weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(th,a)) = sigmoid(x ; a,b) in R
		   epsi                        Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1)
           cascade_type                Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)
           cascade                     Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                       If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
									   If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))
									   cascade(2 , :) reprensent thresholds for each segment
  Outputs
  -------
  
  yest                                 Estimated labels (1 x N) in INT8 format
  fx                                   Additive models (1 x N)

  To compile
  ----------


  mex  -output chlbp_gentleboost_binary_predict_cascade.dll chlbp_gentleboost_binary_predict_cascade.c

  mex  -f mexopts_intel10.bat -output chlbp_gentleboost_binary_predict_cascade.dll chlbp_gentleboost_binary_predict_cascade.c


  Example 1    L_{8;1}^u + L_{4;1}
  ---------

  load viola_24x24
  Ny                                 = 24;
  Nx                                 = 24;
  options.N                          = [8 , 4];
  options.R                          = [1 , 1];
  options.map                        = zeros(2^max(options.N) , length(options.N));
  mapping                            = getmapping(options.N(1),'u2');
  options.map(1:2^options.N(1) , 1)  = mapping.table';
  options.map(1:2^options.N(2) , 2)  = (0:2^options.N(2)-1)';
  options.shiftbox                   = cat(3 , [Ny , Nx ; 1 , 1] , [16 , 16 ; 4 , 4]);


  H                                  = chlbp(X , options);
  figure(1)
  imagesc(H)

  index                              = randperm(length(y));
  y                                  = int8(y);
  options.T                          = 15;

  options.param                      = chlbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options);
  [yest , fx]                        = chlbp_gentleboost_binary_predict_cascade(H , options);
  indp                               = find(y == 1);
  indn                               = find(y ==-1);

  tp                                 = sum(yest(indp) == y(indp))/length(indp)
  fp                                 = 1 - sum(yest(indn) == y(indn))/length(indn)
  er                                 = sum(yest == y)/length(y)

  [dum , ind]                        = sort(y , 'descend');
  figure(2)
  plot(fx(ind))


  Example 2     L_{8;1}^u + L_{4;1} + L_{12;2}^u
  ---------

  load viola_24x24
  Ny                                 = 24;
  Nx                                 = 24;
  options.N                          = [8 , 4 , 12];
  options.R                          = [1 , 1 , 2];
  options.map                        = zeros(2^max(options.N) , length(options.N));
 
  mapping                            = getmapping(options.N(1),'u2');
  options.map(1:2^options.N(1) , 1)  = mapping.table';
  options.map(1:2^options.N(2) , 2)  = (0:2^options.N(2)-1)';
  mapping                            = getmapping(options.N(3),'u2');
  options.map(1:2^options.N(3) , 3)  = mapping.table';
  options.shiftbox                   = cat(3 , [Ny , Nx ; 1 , 1] , [16 , 16 ; 4 , 4] , [Ny , Nx ; 1 , 1]);


  H                                  = chlbp(X , options);
  figure(3)
  imagesc(H)

  index                              = randperm(length(y));
  y                                  = int8(y);
  options.T                          = 100;


  options.param                      = chlbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options);
  [yest , fx]                        = chlbp_gentleboost_binary_predict_cascade(H , options);
  indp                               = find(y == 1);
  indn                               = find(y ==-1);

  tp                                 = sum(yest(indp) == y(indp))/length(indp)
  fp                                 = 1 - sum(yest(indn) == y(indn))/length(indn)
  er                                 = sum(yest == y)/length(y)
 
  [dum , ind]                        = sort(y , 'descend');
  figure(4)
  plot(fx(ind))


  Example 2     L_{8;1}^u + L_{4;1} + L_{12;1.5}^u + + L_{16;2}^u
  ---------

  load viola_24x24
  Ny                                  = 24;
  Nx                                  = 24;
  options.N                           = [8 , 4 , 12 , 16];
  options.R                           = [1 , 1 , 1.5 , 2];
  options.map                         = zeros(2^max(options.N) , length(options.N));
 
  mapping                             = getmapping(options.N(1),'u2');
  options.map(1:2^options.N(1) , 1)   = mapping.table';
  options.map(1:2^options.N(2) , 2)   = (0:2^options.N(2)-1)';
  mapping                             = getmapping(options.N(3),'u2');
  options.map(1:2^options.N(3) , 3)   = mapping.table';
  mapping                             = getmapping(options.N(4),'u2');
  options.map(1:2^options.N(4) , 4)   = mapping.table';

  options.shiftbox                    = cat(3 , [Ny , Nx ; 1 , 1] , [16 , 16 ; 4 , 4] , [Ny , Nx ; 1 , 1] , [Ny , Nx ; 1 , 1]);


  H                                   = chlbp(X , options);
  figure(3)
  imagesc(H)

  index                               = randperm(length(y));
  y                                   = int8(y);
  options.T                           = 100;


  options.param                       = chlbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options);
  [yest , fx]                         = chlbp_gentleboost_binary_predict_cascade(H , options);
  indp                                = find(y == 1);
  indn                                = find(y ==-1);

  tp                                  = sum(yest(indp) == y(indp))/length(indp)
  fp                                  = 1 - sum(yest(indn) == y(indn))/length(indn)
  er                                  = sum(yest == y)/length(y)
 
  [dum , ind]                         = sort(y , 'descend');
  figure(4)
  plot(fx(ind))

 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Reference ""


*/


#include <math.h>
#include <mex.h>

#define sign(a) ((a) >= (0) ? (1) : (-1))

struct opts
{
  int         weaklearner;
  double      epsi;
  int         cascade_type;
  double      *param;
  int          T;
  double      *cascade;
  int          Ncascade;
};

/* Function prototypes */

/*-------------------------------------------------------------------------------------------------------------- */

void  chlbp_gentleboost_binary_predict_cascade(unsigned int * , int , int , struct opts , char *, double *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned int *X ;
	int  Tcascade = 0;
	int *dimsyest;
	struct opts options;
    char *yest;
	double *fx;
	int i , d , N;
	mxArray *mxtemp;
	double *tmp;
	int tempint;
	double  param_default[400]      = {123.000000,19.000000,-1.234925, 0.405912,20.000000, 8.000000, 1.093632,-0.795903,95.000000,20.000000, 0.950404,-0.526123,39.000000, 7.000000, 0.812792,-0.557598,160.000000,10.000000, 0.747027,-0.360020,26.000000, 4.000000, 0.971862,-0.822373,61.000000, 8.000000, 0.679149,-0.414255,131.000000,14.000000,-0.684979, 0.151080,14.000000, 8.000000, 0.617073,-0.376109,92.000000, 7.000000, 0.714601,-0.550345,59.000000,57.000000,-0.529475, 0.249630,141.000000, 6.000000, 0.656193,-0.543811,180.000000, 6.000000, 0.690896,-0.598866,198.000000, 7.000000,-0.725216, 0.112782,17.000000, 1.000000, 0.461032,-0.300761,168.000000, 7.000000, 0.812701,-0.613930,200.000000,20.000000,-0.571839, 0.136212,194.000000,24.000000, 0.439397,-0.172373,101.000000,21.000000,-0.572716, 0.118885,69.000000,15.000000, 0.437576,-0.269506,42.000000, 2.000000, 0.416103,-0.149439,40.000000, 6.000000, 0.449504,-0.303777,103.000000, 8.000000,-0.503342, 0.379690,30.000000, 1.000000, 0.440030,-0.250768,54.000000,12.000000,-0.602087, 0.245752,183.000000, 9.000000, 0.426468,-0.305722,144.000000,12.000000, 0.424041,-0.148955,12.000000,16.000000,-0.845289, 0.080775,202.000000, 8.000000,-0.408533, 0.281680,192.000000, 8.000000, 0.401498,-0.279457, 7.000000, 8.000000, 0.397871,-0.227778,191.000000,19.000000,-0.403195, 0.135068,166.000000, 7.000000,-0.486026, 0.078810,89.000000, 9.000000, 0.368671,-0.299142,36.000000, 9.000000,-0.350489, 0.141661,33.000000, 9.000000, 0.384125,-0.255445,188.000000, 5.000000, 0.743963,-0.692574,148.000000, 7.000000,-0.365869, 0.252057,84.000000, 7.000000, 0.395224,-0.302404,161.000000, 6.000000,-0.471911, 0.086530,90.000000,13.000000, 0.336134,-0.146737,154.000000,10.000000,-0.325079, 0.201435,29.000000, 3.000000, 0.366772,-0.107604,114.000000, 5.000000, 0.732999,-0.669281,98.000000,21.000000,-0.370108, 0.097227,58.000000,28.000000,-0.354058, 0.255729,47.000000, 2.000000, 0.692170,-0.654004,71.000000, 7.000000,-0.400529, 0.349159,22.000000, 6.000000,-0.355369, 0.077474,165.000000,10.000000,-0.322580, 0.211658,133.000000, 8.000000, 0.534231,-0.488533,104.000000, 7.000000, 0.492561,-0.428236,72.000000,10.000000,-0.394152, 0.308416,110.000000, 9.000000,-0.308439, 0.181472,115.000000,18.000000, 0.485466,-0.079214,196.000000,13.000000,-0.387202, 0.079228,24.000000, 5.000000, 0.307231,-0.107417, 4.000000, 7.000000, 0.370797,-0.040615,27.000000, 5.000000, 0.322689,-0.220778,170.000000,10.000000,-0.302749, 0.206939,112.000000, 5.000000, 0.413651,-0.386747,93.000000,13.000000, 0.313595,-0.106537,87.000000,14.000000,-0.308722, 0.117030,64.000000, 7.000000,-0.336211, 0.230845,138.000000,15.000000, 0.353205,-0.108655,60.000000, 3.000000, 0.739144,-0.706684,15.000000, 5.000000, 0.302232,-0.220226,56.000000, 1.000000, 0.349453,-0.275008,37.000000, 2.000000, 0.332515,-0.169944,43.000000, 7.000000,-0.340515, 0.089095,136.000000,10.000000, 0.331548,-0.276701,109.000000,15.000000, 0.395324,-0.051069,100.000000, 7.000000,-0.290074, 0.221041,132.000000,10.000000, 0.330252,-0.188815, 8.000000, 2.000000, 0.321528,-0.043105,86.000000, 5.000000, 0.256124,-0.060966,150.000000, 4.000000,-0.350122, 0.143030,182.000000, 2.000000, 0.364845,-0.310859,44.000000, 4.000000, 0.327946,-0.247483, 2.000000,11.000000,-0.341639, 0.094998,53.000000, 5.000000,-0.285204, 0.113478,134.000000, 8.000000, 0.521234,-0.060925,65.000000, 4.000000,-0.282592, 0.163376,102.000000, 9.000000,-0.659227,-0.022564,35.000000, 7.000000,-0.394838, 0.036525,82.000000,34.000000,-0.525317, 0.026239,146.000000,19.000000, 0.263005,-0.118806,85.000000,16.000000,-0.282732, 0.122308,135.000000, 8.000000, 0.387093,-0.333228,193.000000, 1.000000, 0.623198,-0.569381,129.000000, 3.000000,-0.293658, 0.216511, 3.000000, 0.000000, 0.280033,-0.215683,173.000000, 6.000000,-0.341844, 0.313910,189.000000,15.000000, 0.364578,-0.066529,155.000000,22.000000,-0.280983, 0.087590,203.000000,16.000000, 0.273130,-0.180259,171.000000,13.000000,-0.330449, 0.275738,164.000000,15.000000, 0.414451,-0.059075,48.000000,13.000000,-0.290299, 0.073271,94.000000,12.000000,-0.303010, 0.071087};	
	
	options.weaklearner  = 0;
	options.epsi         = 1;
	options.cascade_type = 0;
	options.Ncascade     = 1;
	options.T            = 100;

	
    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) !=2) || !mxIsUint32(prhs[0]) )
	{		
		mexErrMsgTxt("X must be (d x N) in UINT32 format");	
	}
	
	X           = (unsigned int *)mxGetData(prhs[0]);
	d           = mxGetM(prhs[0]);
	N           = mxGetN(prhs[0]);
		
	/* Input 2  */

	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )		
	{		
		if(!mxIsStruct(prhs[1]) )
		{
			mexErrMsgTxt("options must be a structure");	
		}
		
		mxtemp                            = mxGetField( prhs[1] , 0, "param" );
		if(mxtemp != NULL)
		{	
			if (mxGetM(mxtemp) != 4)
			{		
				mexErrMsgTxt("param must be (4 x T) matrix");	
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

		mxtemp                            = mxGetField(prhs[1] , 0 , "weaklearner");		
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];	
			if(tempint < 0)
			{
				mexErrMsgTxt("weaklearner ={0,1}, force default to 0");	
				options.weaklearner       = 0;	
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
				mexPrintf("cascade_type = {0,1}, force to 0");	
				options.cascade_type      = 0;	
			}
			else
			{	
				options.cascade_type     = tempint;	
			}			
		}	

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade" );
		if(mxtemp != NULL)
		{	
			if(mxGetM(mxtemp) != 2)
			{
				mexErrMsgTxt("cascade must be (2 x Ncascade)");
			}

			options.cascade                           = mxGetPr(mxtemp);
			options.Ncascade                          = mxGetN(mxtemp);

			for(i = 0 ; i < 2*options.Ncascade ; i=i+2)
			{
				Tcascade                            += (int) options.cascade[i];
			}

			if(Tcascade > options.T)
			{
				mexErrMsgTxt("sum(cascade(1 , :)) <= T");
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

			options.cascade                           = (double *)mxMalloc(2*sizeof(double));		
			options.cascade[0]                        = (double) options.T;
			options.cascade[1]                        = 0.0;
		}
	}
		
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
		
	chlbp_gentleboost_binary_predict_cascade(X , d , N , options , yest , fx );
	
	/*------------------------ Free Memory ----------------------------*/

	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "param" )) == NULL )
		{
			mxFree(options.param);
		}

		if ( (mxGetField( prhs[1] , 0 , "cascade" )) == NULL )
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
void  chlbp_gentleboost_binary_predict_cascade(unsigned int *X , int d , int N , struct opts options  , char *yest, double *fx)
{
	double *param = options.param , *cascade = options.cascade;
	int Ncascade = options.Ncascade , weaklearner = options.weaklearner , cascade_type = options.cascade_type;
	double epsi = options.epsi;
	int t , n , c  , Tc;
	int indd , indc , indm , featureIdx;
	double  th , a , b , thresc;
	double sum , sum_total ;
	unsigned int z;
	
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
				th         = param[1 + indm];
				a          = param[2 + indm];
				b          = param[3 + indm];
				z          = X[featureIdx + indd];
				
				if(weaklearner == 0) /* Decision Stump */
				{
/*					sum       += (a*( X[featureIdx + indd]>th ) + b);	 */
					sum       += (a*( z>th ) + b);	
				}
				if(weaklearner == 1) /* Perceptron */
				{	
					sum       += ((2.0/(1.0 + exp(-2.0*epsi*(a*z + b)))) - 1.0);	
/*					sum       += ((2.0/(1.0 + exp(-2.0*epsi*(a*X[featureIdx - 1 + indd] + b)))) - 1.0);	 */
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
		else if (cascade_type == 0)
		{
			fx[n]       = sum;
			yest[n]     = sign(sum);	
		}
		indd       += d;
	}	
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
