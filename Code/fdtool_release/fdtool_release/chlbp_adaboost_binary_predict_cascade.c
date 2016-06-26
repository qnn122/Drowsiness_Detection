/*

  Predict data label with a Strong Classifier trained with chlbp_adaboost_binary_train_cascade

  Usage
  ------

  [yest , fx] = chlbp_adaboost_binary_predict_cascade(X , [options]);
  
  Inputs
  -------

  X                                    Features matrix (d x N) in INT32 format
  options

          param                        Trained param structure (4 x T)
          weaklearner                  Choice of the weak learner used in the training phase
			                           weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
          cascade_type                 Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade
          cascade                      Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                       If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
									   If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))
									   cascade(2 , :) reprensent thresholds for each segment
  Outputs
  -------
  
  yest                                 Estimated labels (1 x N) in INT8 format
  fx                                   Additive models (1 x N)


  To compile
  ----------

  mex  -output chlbp_adaboost_binary_predict_cascade.dll chlbp_adaboost_binary_predict_cascade.c
  mex  -f mexopts_intel10.bat -output chlbp_adaboost_binary_predict_cascade.dll chlbp_adaboost_binary_predict_cascade.c


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


  H                                 = chlbp(X , options);
  figure(1)
  imagesc(H)

  index                             = randperm(length(y));
  y                                 = int8(y);
  options.T                         = 15;

  options.param                     = chlbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);
  [yest , fx]                       = chlbp_adaboost_binary_predict_cascade(H , options);
  indp                              = find(y == 1);
  indn                              = find(y ==-1);

  tp                                = sum(yest(indp) == y(indp))/length(indp)
  fp                                = 1 - sum(yest(indn) == y(indn))/length(indn)
  er                                = sum(yest == y)/length(y)

  [dum , ind]                       = sort(y , 'descend');
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


  options.param                      = chlbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);
  [yest , fx]                        = chlbp_adaboost_binary_predict_cascade(H , options);
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


  options.param                       = chlbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);
  [yest , fx]                         = chlbp_adaboost_binary_predict_cascade(H , options);
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
  int          weaklearner;
  int          cascade_type;
  double      *param;
  int          T;
  double      *cascade;
  int          Ncascade;
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void  chlbp_adaboost_binary_predict_cascade(unsigned int * , int , int , struct opts , char *, double *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned int *X ;		
	int Tcascade = 0;
	int *dimsyest;
	struct opts options;
    char *yest;
	double *fx;
	double  param_default[400]      = {123.000000,17.000000,-0.757073, 0.000000,20.000000,10.000000, 0.571187, 0.000000,95.000000,26.000000, 0.489396, 0.000000,160.000000,12.000000, 0.357342, 0.000000,39.000000, 7.000000, 0.386237, 0.000000, 4.000000, 5.000000, 0.349589, 0.000000,140.000000,10.000000, 0.269589, 0.000000,194.000000,26.000000, 0.299142, 0.000000,26.000000, 7.000000, 0.278204, 0.000000,141.000000,11.000000, 0.305682, 0.000000,24.000000, 5.000000, 0.272224, 0.000000,144.000000,10.000000, 0.217764, 0.000000, 7.000000,13.000000, 0.235855, 0.000000,180.000000, 8.000000, 0.219083, 0.000000,42.000000, 3.000000, 0.218915, 0.000000,30.000000, 1.000000, 0.204764, 0.000000,14.000000,11.000000, 0.215124, 0.000000,61.000000, 9.000000, 0.191655, 0.000000,104.000000,16.000000, 0.222398, 0.000000,17.000000, 1.000000, 0.183161, 0.000000,181.000000,18.000000, 0.209086, 0.000000,33.000000,10.000000, 0.200810, 0.000000,192.000000,14.000000, 0.214840, 0.000000,49.000000, 4.000000, 0.167292, 0.000000,29.000000, 3.000000, 0.202132, 0.000000,13.000000, 4.000000, 0.134771, 0.000000,69.000000,16.000000, 0.153531, 0.000000,101.000000,16.000000,-0.181209, 0.000000,31.000000, 5.000000, 0.167809, 0.000000,137.000000,13.000000, 0.147874, 0.000000,146.000000,14.000000, 0.154115, 0.000000,135.000000,11.000000, 0.125353, 0.000000,90.000000,13.000000, 0.146502, 0.000000,202.000000,10.000000,-0.168766, 0.000000, 8.000000, 1.000000, 0.140291, 0.000000,168.000000,20.000000, 0.156091, 0.000000,200.000000,16.000000,-0.174762, 0.000000,131.000000,12.000000,-0.135919, 0.000000,56.000000, 3.000000, 0.104505, 0.000000,143.000000,16.000000, 0.148801, 0.000000, 3.000000, 1.000000, 0.113145, 0.000000,115.000000,17.000000, 0.124074, 0.000000,191.000000,18.000000,-0.123551, 0.000000,40.000000, 6.000000, 0.145388, 0.000000,103.000000, 8.000000,-0.147863, 0.000000,93.000000, 9.000000, 0.149411, 0.000000, 5.000000, 8.000000,-0.127218, 0.000000,84.000000, 9.000000, 0.122910, 0.000000,148.000000, 8.000000,-0.148932, 0.000000,27.000000, 4.000000, 0.086159, 0.000000,18.000000, 7.000000, 0.153029, 0.000000,136.000000, 9.000000, 0.112010, 0.000000,74.000000, 9.000000,-0.130788, 0.000000,60.000000,10.000000, 0.127055, 0.000000,59.000000,63.000000,-0.153580, 0.000000,92.000000,20.000000, 0.158693, 0.000000,44.000000, 4.000000, 0.129031, 0.000000,165.000000,12.000000,-0.121109, 0.000000,32.000000, 6.000000, 0.107164, 0.000000,114.000000,12.000000, 0.115955, 0.000000,51.000000, 2.000000, 0.089033, 0.000000,58.000000,80.000000, 0.134795, 0.000000,155.000000,22.000000,-0.133790, 0.000000,188.000000,11.000000, 0.131699, 0.000000,36.000000, 4.000000,-0.108341, 0.000000,150.000000, 5.000000,-0.129680, 0.000000,86.000000, 5.000000, 0.142395, 0.000000,166.000000, 6.000000,-0.099503, 0.000000,182.000000, 5.000000, 0.101465, 0.000000,77.000000, 9.000000,-0.082196, 0.000000,11.000000, 7.000000, 0.107064, 0.000000,110.000000, 7.000000,-0.072848, 0.000000,37.000000, 3.000000, 0.106226, 0.000000,107.000000,35.000000, 0.104851, 0.000000,112.000000, 6.000000, 0.119340, 0.000000,12.000000,11.000000,-0.086579, 0.000000,198.000000, 1.000000,-0.094375, 0.000000,109.000000, 6.000000, 0.078661, 0.000000,147.000000, 8.000000,-0.106835, 0.000000,48.000000,12.000000,-0.105151, 0.000000,157.000000,10.000000,-0.093485, 0.000000,183.000000,10.000000, 0.100002, 0.000000,203.000000,41.000000, 0.126534, 0.000000,47.000000, 3.000000, 0.109574, 0.000000,151.000000,11.000000,-0.100517, 0.000000,189.000000,16.000000, 0.066602, 0.000000,133.000000, 9.000000, 0.087938, 0.000000,138.000000,19.000000, 0.098880, 0.000000,127.000000, 9.000000, 0.087811, 0.000000,124.000000,19.000000, 0.095822, 0.000000,161.000000, 6.000000,-0.090063, 0.000000,54.000000, 8.000000,-0.058529, 0.000000,68.000000, 7.000000, 0.067055, 0.000000,87.000000, 6.000000,-0.092150, 0.000000,98.000000,21.000000,-0.095796, 0.000000,15.000000, 8.000000, 0.093939, 0.000000,22.000000, 2.000000,-0.075293, 0.000000,174.000000, 6.000000, 0.090673, 0.000000,164.000000,15.000000, 0.104560, 0.000000,196.000000,11.000000,-0.101679, 0.000000};	
	int i , d , N;
	mxArray *mxtemp;
	double *tmp;
	int tempint;

	options.weaklearner  = 2;
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
			if((tempint < 2) || (tempint > 3))
			{				
				mexPrintf("weaklearner ={2}, force default to 2");			
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
			options.param                             = (double *)mxMalloc(400*sizeof(double));	
			for(i = 0 ; i < 400 ; i++)
			{
				options.param[i]                      = param_default[i];	
			}	
			options.T                                 = 100;

			options.cascade                           = (double *)mxMalloc(2*sizeof(double));		
			options.cascade[0]                        = (double) options.T;
			options.cascade[1]                        = 0.0;
		}
	}
		
	/*------------------------ Output ----------------------------*/

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
	
	chlbp_adaboost_binary_predict_cascade(X , d , N , options , yest , fx );

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
void  chlbp_adaboost_binary_predict_cascade(unsigned int *X , int d , int N , struct opts options  , char *yest, double *fx)
{
	double *param = options.param , *cascade = options.cascade;
	int Ncascade = options.Ncascade , weaklearner = options.weaklearner , cascade_type = options.cascade_type;
	int t , n , c  , Tc;
	int indd , indc , indm , featureIdx;
	double  th , a  , thresc;		
	double sum , sum_total ;
	
	if(weaklearner == 2) /* Decision Stump */
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
					th         = param[1 + indm];
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
			if(cascade_type  == 1)	
			{
				fx[n]       = sum_total;				
				yest[n]     = sign(sum_total);
			}
			else if (cascade_type  == 0)
			{
				fx[n]       = sum;
				yest[n]     = sign(sum);	
			}		
			indd       += d;
		}	
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
