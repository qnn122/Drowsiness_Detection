
/*

  Eval Haar features cascade trained model on a set of image X

  Usage
  ------

  [fx , y]       = eval_haar(X , model);

  
  Inputs
  -------

  I                                     Input images (Ny x Nx x N) in UINT8 format.
  
  model                                 Trained model structure
             weaklearner                Choice of the weak learner used in the training phase (default weaklearner = 2)
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
             param                      Trainned classfier parameters matrix (4 x T). Each row corresponds to :
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity when weaklearner = 2)
			                            b                         Offset (1 x T) (when weaklearner = 2, b = 0)
             dimsItraining              Size of the train images used in the haar computation, i.e. (ny x nx )
             rect_param                 Features rectangles parameters (10 x nR), where nR is the total number of rectangles for the patterns.
                                        (default Vertical(2 x 1) [1 ; -1] and Horizontal(1 x 2) [-1 , 1] patterns) 
										rect_param(: , i) = [ip ; wp ; hp ; nrip ; nr ; xr ; yr ; wr ; hr ; sr], where
										ip     index of the current pattern. ip = [1,...,nP], where nP is the total number of patterns
										wp     width of the current pattern
										hp     height of the current pattern
										nrip   total number of rectangles for the current pattern ip
										nr     index of the current rectangle of the current pattern, nr=[1,...,nrip]
										xr,yr  top-left coordinates of the current rectangle of the current pattern
										wr,hr  width and height of the current rectangle of the current pattern
										sr     weights of the current rectangle of the current pattern 
             F                          Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle
             cascade_type               Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade
             cascade                    Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
										If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))						
										cascade(2 , :) reprensent thresholds for each segment
             standardize                Standardize Input Images 1 = yes, 0 = no (default = 1)


  Outputs
  -------

  fx                                    Output (1 x V) of the last stage/Strong classifier (cascade_type = 0/1) 
  y                                     Ispassing cascade vector(1 x V), y=1 if yes, -1 otherwise 

  To compile
  ----------

  mex  -output eval_haar.dll eval_haar.c

  mex  -g -output eval_haar.dll eval_haar.c

  mex  -f mexopts_intel10.bat -output eval_haar.dll eval_haar.c

 load viola_24x24
 load temp_model1

 fx           = eval_haar(X(:,:,1) , model);



  Example 1    Viola-Jones database
  ---------


  load viola_24x24.mat
  load model_detector_haar_24x24.mat
  thresh           = 0;

  indp             = find(y == 1);
  indn             = find(y ==-1);


  [fx , yfx]       = eval_haar(X , model);
  yest             = int8(sign(fx - thresh));


  tp               = sum(yest(indp) == y(indp))/length(indp)
  fp               = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf             = sum(yest == y)/length(y)
  [tpp1 , fpp1 ]   = basicroc(y , fx);


  model.cascade    = [5 , 10 , 20 , 30 , 35 ; -1.5 ,  -0.75 ,  -0.5 , -0.25 , 0];
  [fx_cascade , y_cascade]       = eval_haar(X , model);
  yest             = int8(sign(fx_cascade));

  tp               = sum(yest(indp) == y(indp))/length(indp)
  fp               = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf             = sum(yest == y)/length(y)
  [tpp2 , fpp2 ]   = basicroc(y , fx_cascade);


  figure(1)
  plot(1:length(y) , fx , 'r' , 1:length(y) , fx_cascade , 'b')
  
  figure(2)
  plot(fpp1 , tpp1 , fpp2 , tpp2 , 'r')
  axis([-0.02 , 1.02 , -0.02 , 1.02])
  legend('No Cascade' , 'Cascade')
  title('HAAR')


  Example 2    Viola-Jones database : incorporing more Features pattern
  ---------

  clear
  load viola_24x24.mat
  load model_detector_haar_24x24.mat
  thresh           = 0;

  indp             = find(y == 1);
  indn             = find(y ==-1);



  fx1              = eval_haar(X , model);
  yest             = int8(sign(fx1 - thresh));


  tp               = sum(yest(indp) == y(indp))/length(indp)
  fp               = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf             = sum(yest == y)/length(y)
  [tpp1 , fpp1 ]   = basicroc(y , fx1);


  load model_detector_haar_24x24_wl2_ct0_nP19.mat

  fx2              = eval_haar(X , model);
  yest             = int8(sign(fx2 - thresh));



  tp               = sum(yest(indp) == y(indp))/length(indp)
  fp               = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf             = sum(yest == y)/length(y)
  [tpp2 , fpp2 ]   = basicroc(y , fx2);


  figure(1)
  plot(1:length(y) , fx1 , 'b' , 1:length(y) , fx2 , 'r')
  legend('2 patterns' , '19 patterns')

  
  figure(2)
  plot(fpp1 , tpp1 , fpp2 , tpp2 , 'r')
  axis([-0.02 , 1.02 , -0.02 , 1.02])
  legend('2 patterns' , '19 patterns')
  title('HAAR')


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 02/20/2009

 Reference ""


*/


#include <math.h>
#include <mex.h>

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif

#define sign(a)    ((a) >= (0) ? (1.0) : (-1.0))
 
struct model
{
	int            weaklearner;
	double         epsi;
	double        *param;
	int            T;
	double        *dimsItraining;
	int            ny;
	int            nx;
	double        *rect_param;
	int            nR;
	unsigned int  *F;
	int            nF;
	double        *cascade;
	int            Ncascade;
	int            cascade_type;
	int            standardize;
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
unsigned int Area(unsigned int * , int , int , int , int , int );
double haar_feat(unsigned int *  , int  , double * , unsigned int * , int , int , int );
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
void eval_haar(unsigned char * , int , int , int , struct model , double * , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned char *I;
	struct model detector;   
    const int *dimsI ;
    int numdimsI , Tcascade = 0;
    double *fx , *y;
	double  param_default[400]      = {10992.0000,-6.1626,-0.7788,0.0000,76371.0000,24.2334,0.6785,0.0000,4623.0000,-0.6328,-0.5489,0.0000,58198.0000,-4.4234,-0.5018,0.0000,67935.0000,-12.8916,-0.4099,0.0000,19360.0000,-1.7222,0.1743,0.0000,60243.0000,-1.9187,-0.2518,0.0000,3737.0000,-0.9260,0.1791,0.0000,58281.0000,-16.1455,-0.2447,0.0000,13245.0000,-1.6818,0.2183,0.0000,4459.0000,1.7972,0.2043,0.0000,7765.0000,3.0506,0.1665,0.0000,10105.0000,-2.0763,0.1764,0.0000,2301.0000,-2.4221,-0.1526,0.0000,4250.0000,-0.2044,0.1077,0.0000,59328.0000,24.8328,0.2129,0.0000,10127.0000,-2.1996,0.1746,0.0000,65144.0000,-35.6228,-0.2307,0.0000,43255.0000,-0.5288,0.1970,0.0000,57175.0000,-0.2119,0.0597,0.0000,59724.0000,-27.5468,-0.2059,0.0000,13278.0000,-2.1100,0.1895,0.0000,55098.0000,22.4124,0.1913,0.0000,13238.0000,-1.7093,0.1707,0.0000,62386.0000,0.3067,0.1283,0.0000,24039.0000,6.9595,0.1639,0.0000,43211.0000,-0.5982,0.1188,0.0000,62852.0000,9.6709,0.1652,0.0000,43236.0000,-0.6296,0.1530,0.0000,45833.0000,1.7152,0.1974,0.0000,7095.0000,-1.2430,0.1269,0.0000,76347.0000,-27.4002,-0.1801,0.0000,3737.0000,-0.8826,0.1462,0.0000,65143.0000,36.2253,0.1581,0.0000,13160.0000,-2.5302,0.1469,0.0000,4845.0000,0.7053,-0.0690,0.0000,52810.0000,-13.5220,-0.1594,0.0000,43234.0000,0.5907,-0.1420,0.0000,60847.0000,39.2252,0.1563,0.0000,43234.0000,-0.5423,0.1417,0.0000,56659.0000,0.7945,0.1387,0.0000,56930.0000,-1.3875,-0.1496,0.0000,13224.0000,-1.5798,0.1080,0.0000,63154.0000,14.8166,0.1961,0.0000,13162.0000,-2.3354,0.1639,0.0000,10722.0000,0.6559,0.2141,0.0000,7528.0000,1.1026,0.1077,0.0000,4263.0000,0.1324,-0.0485,0.0000,45151.0000,-1.4198,-0.1234,0.0000,7095.0000,1.5141,-0.1367,0.0000,68446.0000,-25.0890,-0.1744,0.0000,43277.0000,-0.5919,0.1564,0.0000,3613.0000,-0.5823,-0.1439,0.0000,5418.0000,3.9535,-0.1502,0.0000,58985.0000,24.7405,0.1754,0.0000,43785.0000,-0.9376,0.1194,0.0000,46582.0000,-5.8589,-0.1286,0.0000,43470.0000,-0.6392,0.1396,0.0000,10262.0000,-2.9209,-0.1251,0.0000,10105.0000,-2.0250,0.0960,0.0000,3555.0000,0.7341,0.1348,0.0000,10115.0000,1.6321,-0.1274,0.0000,76579.0000,-39.8316,-0.1442,0.0000,10228.0000,1.8771,-0.1245,0.0000,57005.0000,2.3937,0.1431,0.0000,43830.0000,-0.7996,0.0652,0.0000,48673.0000,8.8965,0.1181,0.0000,18845.0000,-2.2572,0.0872,0.0000,50225.0000,-1.5850,-0.1181,0.0000,43284.0000,0.5782,-0.1278,0.0000,72000.0000,-8.5961,-0.1282,0.0000,43214.0000,-0.6367,0.1053,0.0000,72559.0000,23.7860,0.1368,0.0000,43792.0000,1.0846,-0.1150,0.0000,56537.0000,-0.1965,-0.1262,0.0000,13421.0000,8.9499,0.0433,0.0000,172.0000,0.5319,-0.0946,0.0000,68220.0000,20.5078,0.1688,0.0000,16105.0000,-1.8842,0.1081,0.0000,79153.0000,5.8776,0.1301,0.0000,19180.0000,2.0606,0.1314,0.0000,13.0000,0.5438,-0.0802,0.0000,67201.0000,-6.6425,-0.1443,0.0000,43210.0000,0.5881,-0.1349,0.0000,65075.0000,-44.1279,-0.1170,0.0000,43214.0000,0.5392,-0.0840,0.0000,139.0000,0.2841,0.1480,0.0000,10209.0000,1.8835,-0.0957,0.0000,44409.0000,-1.0357,-0.1648,0.0000,43210.0000,-0.5252,0.1002,0.0000,47431.0000,6.8252,0.0927,0.0000,10235.0000,-1.3325,0.0795,0.0000,14896.0000,-12.2989,-0.0802,0.0000,1752.0000,-0.8487,-0.1193,0.0000,6964.0000,-1.7033,0.0944,0.0000,64124.0000,32.1583,0.1058,0.0000,43215.0000,-0.6988,0.0997,0.0000,76579.0000,-39.5722,-0.0932,0.0000,43966.0000,1.0216,-0.0926,0.0000,68446.0000,-25.5175,-0.1295,0.0000};
	double	rect_param_default[40]  = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	mxArray *mxtemp;
    int i , Ny , Nx , V = 1  , tempint;
	double *tmp;
	
	
	detector.weaklearner  = 0; 
	detector.epsi         = 0.1;
    detector.nR           = 4;
	detector.cascade_type = 0;
	detector.Ncascade     = 1;
	detector.standardize  = 1;

    if ((nrhs < 2))       
    {		
        mexErrMsgTxt("At least 2 inputs are requiered for detector");	
	}
	
    /* Input 1  */

    dimsI                               = mxGetDimensions(prhs[0]);
    numdimsI                            = mxGetNumberOfDimensions(prhs[0]);
    
    if( mxIsEmpty(prhs[0]) || !mxIsUint8(prhs[0]) )
    {        
        mexErrMsgTxt("I must be (Ny x Nx x N) in UINT8 format");   
    }

    Ny          = dimsI[0];  
    Nx          = dimsI[1];

	if(numdimsI > 2)
	{
		V       = dimsI[2];
	}
    
    I                                     = (unsigned char *)mxGetData(prhs[0]); 
 
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
				mexPrintf("weaklearner = {0,1,2}, force to 2");		
				detector.weaklearner      = 2;
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

			if ((Ny != detector.ny ) || (Nx != detector.nx ))       
			{
				mexErrMsgTxt("I must be  ny x nx");		
			}	
		}

		mxtemp                             = mxGetField( prhs[1] , 0, "rect_param" );
		if(mxtemp != NULL)
		{
			detector.rect_param            = mxGetPr(mxtemp);              
			detector.nR                    = mxGetN(mxtemp);;				
		}
		else
		{
			detector.rect_param            = (double *)mxMalloc(40*sizeof(double));	
			for (i = 0 ; i < 40 ; i++)
			{		
				detector.rect_param[i]     = rect_param_default[i];
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
			detector.nF                    = number_haar_features(Ny , Nx , detector.rect_param , detector.nR);
			detector.F                     = (unsigned int *)mxMalloc(5*detector.nF*sizeof(unsigned int));
			haar_featlist(Ny , Nx , detector.rect_param , detector.nR , detector.F);	
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade_type" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 3))
			{
				mexPrintf("cascade_type = {0,1,2}, force to 2");			
				detector.cascade_type     = 2;	
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
				Tcascade                  += (int) detector.cascade[i];
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

		mxtemp                            = mxGetField( prhs[1] , 0, "standardize" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];
			
			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("standardize = {0,1}, force to 1");		
				detector.standardize      = 1;
			}
			else
			{
				detector.standardize      = tempint;	
			}			
		}
    }
	else	
	{	
		detector.param                 = (double *)mxMalloc(400*sizeof(double));	
		for(i = 0 ; i < 400 ; i++)
		{
			detector.param[i]          = param_default[i];	
		}	
		detector.T                     = 10;

		detector.rect_param            = (double *)mxMalloc(40*sizeof(double));	
		for (i = 0 ; i < 40 ; i++)
		{		
			detector.rect_param[i]     = rect_param_default[i];
		}			

		detector.nF                    = number_haar_features(Ny , Nx , detector.rect_param , detector.nR);
		detector.F                     = (unsigned int *)mxMalloc(5*detector.nF*sizeof(unsigned int));
		haar_featlist(Ny , Nx , detector.rect_param , detector.nR , detector.F);	

		detector.cascade                = (double *)mxMalloc(2*sizeof(double));
		detector.cascade[0]             = (double) detector.T;
		detector.cascade[1]             = 0.0;
		detector.Ncascade               = 1;
	}
    
    /*------------------------ Output ----------------------------*/
 
    plhs[0]                    = mxCreateDoubleMatrix(1 , V , mxREAL);
    fx                         = mxGetPr(plhs[0]);

    plhs[1]                    = mxCreateDoubleMatrix(1 , V , mxREAL);
    y                          = mxGetPr(plhs[1]);

    /*------------------------ Main Call ----------------------------*/
	
	eval_haar(I , Ny , Ny , V , detector  , fx , y);
	
	/*--------------------------- Free memory -----------------------*/
	
	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "param" )) == NULL )
		{
			mxFree(detector.param);
		}
		if ( (mxGetField( prhs[1] , 0 , "rect_param" )) == NULL )
		{
			mxFree(detector.rect_param);
		}
		if ( (mxGetField( prhs[1] , 0 , "F" )) == NULL )
		{
			mxFree(detector.F);
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
		mxFree(detector.rect_param);
	}	
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void eval_haar(unsigned char *I , int Ny , int Nx , int V , struct model detector , double *fx , double *y)			   
{
    double   *param = detector.param , *rect_param = detector.rect_param , *cascade = detector.cascade;
    unsigned int  *II , *Itemp , tempI;
	unsigned int*F = detector.F;
	int weaklearner = detector.weaklearner , Ncascade = detector.Ncascade;
	int nR = detector.nR , nF = detector.nF , cascade_type = detector.cascade_type , standardize = detector.standardize;
	double epsi = detector.epsi;	
	double z , sum , sum_total , a , b , th , thresc;
	int i , v , c , f , Tc , NyNx = Ny*Nx , indNyNx = 0 , indf , indc  , idxF , last = NyNx - 1;
	double  var  , mean , std , cteNyNx = 1.0/NyNx;
	
	II                   = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Itemp                = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	
	if(standardize)
	{
		for(v = 0 ; v < V ; v++)
		{
			MakeIntegralImage(I + indNyNx , II , Nx , Ny  , Itemp);
			var           = 0.0;
			y[v]          = 1.0;
			
			for(i = 0 ; i < NyNx ; i++)
			{				
				tempI      = I[i + indNyNx];		
				var       += (tempI*tempI);	
			}
			
			var          *= cteNyNx;
			mean          = II[last]*cteNyNx;
			std           = 1.0/sqrt(var - mean*mean);

			indf          = 0;
			indc          = 0;
			sum_total     = 0.0;
			for (c = 0 ; c < Ncascade ; c++)
			{		
				Tc     = (int) cascade[0 + indc];	
				thresc = cascade[1 + indc];
				sum    = 0.0;
				for (f = 0 ; f < Tc ; f++)
				{
					idxF  = ((int) param[0 + indf] - 1);	
					z     = haar_feat(II , idxF , rect_param , F , Ny , nR , nF);
					
					th    =  param[1 + indf];
					a     =  param[2 + indf];
					b     =  param[3 + indf];
					
					if(weaklearner == 0)						
					{					
						sum    += (a*( (z*std) > th ) + b);	
					}			
					else if(weaklearner == 1)
					{
						sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a*(z*std) + b)))) - 1.0);	
					}
					else if(weaklearner == 2)
					{	
						sum    += a*sign((z*std) - th);	
					}	
					indf      += 4;
				}			
				sum_total     += sum;
	
				if((sum_total < thresc) && (cascade_type == 1))			
				{
					y[v]  = -1.0;
					break;	
				}
				else if((sum < thresc) && (cascade_type == 0))	
				{
					y[v]  = -1.0;
					break;	
				}			
				indc      += 2; 
			}
			if(cascade_type == 1 )
			{
				fx[v]     = sum_total;	
			}
			else if (cascade_type == 0 )
			{
				fx[v]     = sum;	
			}
			indNyNx  += NyNx;	
		}
	}
	else
	{
		for(v = 0 ; v < V ; v++)
		{		
			MakeIntegralImage(I + indNyNx , II , Nx , Ny  , Itemp);	
			indf          = 0;
			indc          = 0;
			sum_total     = 0.0;
			
			for (c = 0 ; c < Ncascade ; c++)
			{
				Tc     = (int) cascade[0 + indc];	
				thresc = cascade[1 + indc];
				sum    = 0.0;
				
				for (f = 0 ; f < Tc ; f++)
				{			
					idxF  = ((int) param[0 + indf] - 1);

					z     =  haar_feat(II , idxF , rect_param , F , Ny , nR , nF);
					th    =  param[1 + indf];
					a     =  param[2 + indf];
					b     =  param[3 + indf];

					if(weaklearner == 0)				
					{					
						sum    += (a*( z > th ) + b);	
					}			
					else if(weaklearner == 1)
					{
						sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a*z + b)))) - 1.0);	
					}
					else if(weaklearner == 2)
					{			
						sum    += a*sign(z - th);	
					}
					indf      += 4;	
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
				indc      += 2; 
			}		
			if(cascade_type == 1)	
			{
				fx[v]     = sum_total;	
			}
			else if (cascade_type == 0)
			{				
				fx[v]     = sum;	
			}
			indNyNx  += NyNx;	
		}	
	}
	free(II);
	free(Itemp);
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

/*----------------------------------------------------------------------------------------------------------------------------------------------*/
void MakeIntegralImage(unsigned char *pIn, unsigned int *pOut, int iXmax, int iYmax , unsigned int *pTemp)
{
	/* Variable declaration */
	int x , y , indx = 0;
	
	for(x=0 ; x<iXmax ; x++)
	{
		pTemp[indx]     = (unsigned int) pIn[indx];	
		indx           += iYmax;
	}
	for(y = 1 ; y<iYmax ; y++)
	{
		pTemp[y]        = pTemp[y - 1] + (unsigned int)pIn[y];
	}
	pOut[0]             = (unsigned int) pIn[0];
	indx                = iYmax;
	for(x=1 ; x<iXmax ; x++)
	{
		pOut[indx]      = pOut[indx - iYmax] + pTemp[indx];
		indx           += iYmax;
	}
	for(y = 1 ; y<iYmax ; y++)
	{
		pOut[y]         = pOut[y - 1] + (unsigned int) pIn[y];
	}
	
	/* Calculate integral image */

	indx                = iYmax;
	for(x = 1 ; x < iXmax ; x++)
	{
		for(y = 1 ; y < iYmax ; y++)
		{
			pTemp[y + indx]    = pTemp[y - 1 + indx] + (unsigned int) pIn[y + indx];		
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
	if( x==0 )
	{
		return(II[(y+h1) + w1*Ny] - II[y1 + w1*Ny]);	
	}
	if( y==0 )
	{	
		return(II[h1 + (x+w1)*Ny] - II[h1 + x1*Ny]);		
	}
	else
	{	
		return (II[(y+h1) + (x+w1)*Ny] - (II[y1 + (x+w1)*Ny] + II[(y+h1) + x1*Ny]) + II[y1 + x1*Ny]);	
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double haar_feat(unsigned int *II , int featidx , double *rect_param , unsigned int *F , int Ny , int nR , int nF)
{
	int x , xr , y , yr , w , wr , h , hr , r ,  R , indR , indF = featidx*6;
	int coeffw , coeffh;
	double val = 0.0 , s;

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
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
