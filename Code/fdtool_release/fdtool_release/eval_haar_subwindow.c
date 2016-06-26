
/*

  Image eval_haar_subwindow by boosting (Gentle & Ada) 

  Usage
  ------

  [fx , y] = eval_haar_subwindow(I , model);

  
  Inputs
  -------

  I                                     Input image (Ny x Nx) in UINT8 format
  
  model                                 Trainned model structure

             weaklearner                Choice of the weak learner used in the training phase
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
             param                      Trainned classfier parameters matrix (4 x T). Each row corresponds to :
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity when weaklearner = 2)
			                            b                         Offset (1 x T) (when weaklearner = 2, b = 0)
			 dimsItraining              Size of the trainnig images used in the mblbp'model, i.e. (ny x nx) (default ny = 24, nx = 24)
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

  Outputs
  -------
  
  fx                                    Output of the last stage/Strong classifier (cascade_type = 0/1) 
  y                                     Ispassing cascade vector(1 x V), y=1 if yes, -1 otherwise 

  To compile
  ----------


  mex  -output eval_haar_subwindow.dll eval_haar_subwindow.c

  mex  -g -output eval_haar_subwindow.dll eval_haar_subwindow.c


  mex  -f mexopts_intel10.bat -output eval_haar_subwindow.dll eval_haar_subwindow.c

 load viola_24x24
 load temp_model1

 fx    = eval_haar_subwindow(X(:,:,1) , model);



  Example 1
  ---------


  load model_detector_haar_24x24.mat

  model.cascade          = [5 , 10 , 20 , 30 , 35 ; -1.5 ,  -0.75 ,  -0.5 , -0.25 , 0];


  Origx            = 145;
  Origy            = 225;
  size_sub         = 45;


%  Origx            = 275;
%  Origy            = 300;
%  size_sub         = 45;



%  Origx            = 155; %155
%  Origy            = 235; %140
%  size_sub         = 24;

  Origx            = 100;
  Origy            = 60;
  size_sub         = 55;


  Origx            = 145;
  Origy            = 225;
  size_sub         = 45;

  Origx            = 477;
  Origy            = 340;
  size_sub         = 55;

  Origx            = 595;
  Origy            = 395;
  size_sub         = 60;


  I                = (rgb2gray(imread('class57.jpg')));

  im               = I(Origy:Origy + size_sub - 1 , Origx:Origx + size_sub - 1);
  tic,fx           = eval_haar_subwindow(im , model),toc
  tic,fx           = eval_haar_subwindow(imresize(im , [24 , 24]) , model),toc

  model            = rmfield(model , 'cascade');
  tic,fx           = eval_haar_subwindow(im , model),toc
  tic,fx           = eval_haar_subwindow(imresize(im , [24 , 24]) , model),toc


  figure(1)

  imagesc(I)
  hold on
  rectangle('Position' , [Origx , Origy , size_sub , size_sub] ,'Edgecolor',[0,1,0],'LineWidth',2);
  colormap(gray);
  title(sprintf('fx = %6.3f' , fx));
  hold off

  figure(2)
  imagesc(im)
  title(sprintf('fx = %6.3f' , fx));
  colormap(gray)




  Example 2
  ---------

  load model_detector_haar_24x24.mat
  figure(1)
  I                   = (rgb2gray(imread('class57.jpg')));
  Icrop               = (imcrop(I));
  Ires                = imresize(Icrop , [24 , 24]);

  %title(sprintf('fx_{scale} = %6.4f, fx_{interp} = %6.4f' , fx_scale , fx_interp));

  
  fx_scale_single     = eval_haar_subwindow(Icrop , model)
  fx_interp_single    = eval_haar_subwindow(Ires , model )

	
  model.cascade       = [5 , 10 , 20 , 30 , 35 ; -1.5 ,  -0.75 ,  -0.5 , -0.25 , 0];
  model.cascade       = [1 , 2 , 3 , 4 , 10 , 20 , 30 , 30 ; -0.75 ,-0.6 , -0.5, -0.25,  0 ,  0 , 0 , 0];

  fx_scale_cascade    = eval_haar_subwindow(Icrop , model)
  fx_interp_cascade   = eval_haar_subwindow(Ires , model)


  model.cascade_type  = 1;

  fx_scale_multiexit  = eval_haar_subwindow(Icrop , model)
  fx_interp_multiexit = eval_haar_subwindow(Ires , model)

  

  figure(2)
  imagesc(Icrop)
  colormap(gray)
  title(sprintf('fx_{scale-single} = %6.4f' , fx_scale_single ));

  figure(3)
  imagesc(Ires)
  colormap(gray)
  title(sprintf('fx_{interp-single} = %6.4f' , fx_interp_single));


  figure(4)
  imagesc(Icrop)
  colormap(gray)
  title(sprintf('fx_{scale-cascade} = %6.4f' , fx_scale_cascade ));

  figure(5)
  imagesc(Ires)
  colormap(gray)
  title(sprintf('fx_{interp-cascade} = %6.4f' , fx_interp_cascade));


  figure(6)
  imagesc(Icrop)
  colormap(gray)
  title(sprintf('fx_{scale-multiexit} = %6.4f' , fx_scale_multiexit ));

  figure(7)
  imagesc(Ires)
  colormap(gray)
  title(sprintf('fx_{interp-multiexit} = %6.4f' , fx_interp_multiexit));


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
};

/*------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

int Round(double);
int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
void MakeIntegralImage(unsigned char *, unsigned int*, int , int , unsigned int *);
unsigned int Area(unsigned int *  , int , int , int , int , int );
double eval_haar_subwindow(unsigned char * , int , int , struct model , double *);

/*------------------------------------------------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned char *I;	
	struct model detector;
    const int *dimsI ;
    int numdimsI , Tcascade = 0;
	mxArray *mxtemp;
    int i , Ny , Nx  , tempint;
    double *fx , *yfx, *tmp;
	double  param_default[400]      = {10992.0000,-6.1626,-0.7788,0.0000,76371.0000,24.2334,0.6785,0.0000,4623.0000,-0.6328,-0.5489,0.0000,58198.0000,-4.4234,-0.5018,0.0000,67935.0000,-12.8916,-0.4099,0.0000,19360.0000,-1.7222,0.1743,0.0000,60243.0000,-1.9187,-0.2518,0.0000,3737.0000,-0.9260,0.1791,0.0000,58281.0000,-16.1455,-0.2447,0.0000,13245.0000,-1.6818,0.2183,0.0000,4459.0000,1.7972,0.2043,0.0000,7765.0000,3.0506,0.1665,0.0000,10105.0000,-2.0763,0.1764,0.0000,2301.0000,-2.4221,-0.1526,0.0000,4250.0000,-0.2044,0.1077,0.0000,59328.0000,24.8328,0.2129,0.0000,10127.0000,-2.1996,0.1746,0.0000,65144.0000,-35.6228,-0.2307,0.0000,43255.0000,-0.5288,0.1970,0.0000,57175.0000,-0.2119,0.0597,0.0000,59724.0000,-27.5468,-0.2059,0.0000,13278.0000,-2.1100,0.1895,0.0000,55098.0000,22.4124,0.1913,0.0000,13238.0000,-1.7093,0.1707,0.0000,62386.0000,0.3067,0.1283,0.0000,24039.0000,6.9595,0.1639,0.0000,43211.0000,-0.5982,0.1188,0.0000,62852.0000,9.6709,0.1652,0.0000,43236.0000,-0.6296,0.1530,0.0000,45833.0000,1.7152,0.1974,0.0000,7095.0000,-1.2430,0.1269,0.0000,76347.0000,-27.4002,-0.1801,0.0000,3737.0000,-0.8826,0.1462,0.0000,65143.0000,36.2253,0.1581,0.0000,13160.0000,-2.5302,0.1469,0.0000,4845.0000,0.7053,-0.0690,0.0000,52810.0000,-13.5220,-0.1594,0.0000,43234.0000,0.5907,-0.1420,0.0000,60847.0000,39.2252,0.1563,0.0000,43234.0000,-0.5423,0.1417,0.0000,56659.0000,0.7945,0.1387,0.0000,56930.0000,-1.3875,-0.1496,0.0000,13224.0000,-1.5798,0.1080,0.0000,63154.0000,14.8166,0.1961,0.0000,13162.0000,-2.3354,0.1639,0.0000,10722.0000,0.6559,0.2141,0.0000,7528.0000,1.1026,0.1077,0.0000,4263.0000,0.1324,-0.0485,0.0000,45151.0000,-1.4198,-0.1234,0.0000,7095.0000,1.5141,-0.1367,0.0000,68446.0000,-25.0890,-0.1744,0.0000,43277.0000,-0.5919,0.1564,0.0000,3613.0000,-0.5823,-0.1439,0.0000,5418.0000,3.9535,-0.1502,0.0000,58985.0000,24.7405,0.1754,0.0000,43785.0000,-0.9376,0.1194,0.0000,46582.0000,-5.8589,-0.1286,0.0000,43470.0000,-0.6392,0.1396,0.0000,10262.0000,-2.9209,-0.1251,0.0000,10105.0000,-2.0250,0.0960,0.0000,3555.0000,0.7341,0.1348,0.0000,10115.0000,1.6321,-0.1274,0.0000,76579.0000,-39.8316,-0.1442,0.0000,10228.0000,1.8771,-0.1245,0.0000,57005.0000,2.3937,0.1431,0.0000,43830.0000,-0.7996,0.0652,0.0000,48673.0000,8.8965,0.1181,0.0000,18845.0000,-2.2572,0.0872,0.0000,50225.0000,-1.5850,-0.1181,0.0000,43284.0000,0.5782,-0.1278,0.0000,72000.0000,-8.5961,-0.1282,0.0000,43214.0000,-0.6367,0.1053,0.0000,72559.0000,23.7860,0.1368,0.0000,43792.0000,1.0846,-0.1150,0.0000,56537.0000,-0.1965,-0.1262,0.0000,13421.0000,8.9499,0.0433,0.0000,172.0000,0.5319,-0.0946,0.0000,68220.0000,20.5078,0.1688,0.0000,16105.0000,-1.8842,0.1081,0.0000,79153.0000,5.8776,0.1301,0.0000,19180.0000,2.0606,0.1314,0.0000,13.0000,0.5438,-0.0802,0.0000,67201.0000,-6.6425,-0.1443,0.0000,43210.0000,0.5881,-0.1349,0.0000,65075.0000,-44.1279,-0.1170,0.0000,43214.0000,0.5392,-0.0840,0.0000,139.0000,0.2841,0.1480,0.0000,10209.0000,1.8835,-0.0957,0.0000,44409.0000,-1.0357,-0.1648,0.0000,43210.0000,-0.5252,0.1002,0.0000,47431.0000,6.8252,0.0927,0.0000,10235.0000,-1.3325,0.0795,0.0000,14896.0000,-12.2989,-0.0802,0.0000,1752.0000,-0.8487,-0.1193,0.0000,6964.0000,-1.7033,0.0944,0.0000,64124.0000,32.1583,0.1058,0.0000,43215.0000,-0.6988,0.0997,0.0000,76579.0000,-39.5722,-0.0932,0.0000,43966.0000,1.0216,-0.0926,0.0000,68446.0000,-25.5175,-0.1295,0.0000};
	double	rect_param_default[40]  = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
		
	detector.weaklearner  = 2; 
	detector.epsi         = 0.1;
    detector.nR           = 4;
	detector.cascade_type = 0;
	detector.Ncascade     = 1;
	detector.nx           = 24;
	detector.ny           = 24;

    if ((nrhs < 2))       
    {	
        mexErrMsgTxt("At least 2 inputs are requiered for detector");	
	}
	
    /* Input 1  */

    numdimsI             = mxGetNumberOfDimensions(prhs[0]);
    
    if( (numdimsI > 2) && !mxIsUint8(prhs[0]) )
    {
        mexErrMsgTxt("I must be (Ny x Nx) in UINT8 format");   
    }
    
    I           = (unsigned char *)mxGetData(prhs[0]); 
    dimsI       = mxGetDimensions(prhs[0]);
    
    Ny          = dimsI[0];  
    Nx          = dimsI[1];
    
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

		mxtemp                    = mxGetField( prhs[1] , 0, "epsi" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			if(tmp[0] < 0.0 )
			{
				mexPrintf("epsi must be > 0, force to 0.1");		
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
		
		mxtemp                             = mxGetField( prhs[1] , 0, "rect_param" );
		if(mxtemp != NULL)
		{			
			if((mxGetM(mxtemp) != 10) || !mxIsDouble(mxtemp) )
			{		
				mexErrMsgTxt("rect_param must be (10 x nR) in DOUBLE format");	
			}
			
			detector.rect_param            = (double *) mxGetData(mxtemp);
			detector.nR                    = mxGetN(mxtemp);
		}
		else
		{	
			detector.rect_param            = (double *)mxMalloc(40*sizeof(double));	
			for(i = 0 ; i < 40 ; i++)
			{
				detector.rect_param[i]     = rect_param_default[i];	
			}	
		}	

		mxtemp                             = mxGetField( prhs[1] , 0, "F" );
		if(mxtemp != NULL)
		{
			detector.F                     = (unsigned int *)mxGetData(mxtemp);	
			detector.nF                    = mxGetN(mxtemp);
		}
		else
		{
			detector.nF                    = number_haar_features(detector.ny , detector.nx , detector.rect_param , detector.nR);
			detector.F                     = (unsigned int *)mxMalloc(6*detector.nF*sizeof(int));
			haar_featlist(detector.ny , detector.nx , detector.rect_param , detector.nR , detector.F);
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
    }
	else		
	{	
		detector.param                 = (double *)mxMalloc(400*sizeof(double));	
		for(i = 0 ; i < 400 ; i++)
		{
			detector.param[i]          = param_default[i];	
		}	
		detector.T                     = 10;

		detector.nF                    = number_haar_features(detector.ny , detector.nx , detector.rect_param , detector.nR);
		detector.F                     = (unsigned int *)mxMalloc(6*detector.nF*sizeof(int));
		haar_featlist(detector.ny , detector.nx , detector.rect_param , detector.nR , detector.F);

		detector.rect_param            = (double *)mxMalloc(40*sizeof(double));	
		for(i = 0 ; i < 40 ; i++)
		{
			detector.rect_param[i]     = rect_param_default[i];	
		}	

		detector.cascade                = (double *)mxMalloc(2*sizeof(double));
		detector.cascade[0]             = (double) detector.T;
		detector.cascade[1]             = 0.0;
		detector.Ncascade               = 1;
	}
    
    /*------------------------ Output ----------------------------*/

    plhs[0]                    = mxCreateDoubleMatrix(1 , 1 , mxREAL);
    fx                         = mxGetPr(plhs[0]);

    plhs[1]                    = mxCreateDoubleMatrix(1 , 1 , mxREAL);
    yfx                        = mxGetPr(plhs[1]);


    /*------------------------ Main Call ----------------------------*/
	
    fx[0]                      = eval_haar_subwindow(I , Ny , Nx  , detector , yfx);
		
    /*----------------------- Outputs -------------------------------*/

	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "param" )) == NULL )
		{
			mxFree(detector.param);
		}
		if ( (mxGetField( prhs[1] , 0 , "rect_param" )) == NULL)
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
		mxFree(detector.rect_param);        
		mxFree(detector.cascade);
		mxFree(detector.F);        
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double eval_haar_subwindow(unsigned char *I , int Ny , int Nx  , struct model detector , double *yfx)
{
	double  *cascade = detector.cascade, *param = detector.param , *rect_param = detector.rect_param;
	int Ncascade = detector.Ncascade , ny = detector.ny , nx = detector.nx , cascade_type = detector.cascade_type , weaklearner = detector.weaklearner ;
	double epsi  = detector.epsi , sum , sum_total = 0.0, a , b , th , thresh , fx ;	
	int i, NxNy = Ny*Nx , last = NxNy - 1;
	int c , f , Tc , indc = 0 , indf = 0, idxF ,  x , xr , y , yr , w , wr , h , hr  , r   , R , indR , coeffw , coeffh;
	unsigned int *F = detector.F;
	double z  , s , var = 0.0 , mean , std , cteNxNy = 1.0/NxNy , scalex = (Nx - 0 )/(double)nx , scaley = (Ny - 0 )/(double)ny ;
	double ctescale = 1.0/(scalex*scaley);
	unsigned int *II  , *Itemp , tempI;

	II                   = (unsigned int *) malloc(NxNy*sizeof(unsigned int));
	Itemp                = (unsigned int *) malloc(NxNy*sizeof(unsigned int));

	MakeIntegralImage(I , II , Nx , Ny , Itemp);

	for(i = 0 ; i < NxNy ; i++)
	{
		tempI      = I[i];		
		var       += (tempI*tempI);	
	}
				
	var      *= cteNxNy;
	mean      = II[last]*cteNxNy;
	std       = ctescale/sqrt(var - mean*mean);
/*	std       = 1.0/sqrt(var - mean*mean); */

	yfx[0]    = 1.0;

	for (c = 0 ; c < Ncascade ; c++)
	{
		Tc     = (int) cascade[0 + indc];
		thresh = cascade[1 + indc];	
		sum    = 0.0;
		
		for (f = 0 ; f < Tc ; f++)
		{
			idxF  = ((int) param[0 + indf] - 1)*6;
			th    =  param[1 + indf];
			a     =  param[2 + indf];
			b     =  param[3 + indf];
			x     = F[1 + idxF];
			y     = F[2 + idxF];
			w     = F[3 + idxF];
			h     = F[4 + idxF];

			indR  = F[5 + idxF];
			R     = (int) rect_param[3 + indR];
			
			z     = 0.0;
			for (r = 0 ; r < R ; r++)
			{
				coeffw  = w/(int)rect_param[1 + indR];		
				coeffh  = h/(int)rect_param[2 + indR];
				xr      = Round(scalex*(x + (coeffw*(int)rect_param[5 + indR])));
				yr      = Round(scaley*(y + (coeffh*(int)rect_param[6 + indR])));
				wr      = Round(scalex*(coeffw*(int)(rect_param[7 + indR])));
				hr      = Round(scaley*(coeffh*(int)(rect_param[8 + indR])));
				s       = rect_param[9 + indR];
				z      += s*Area(II , xr  , yr  , wr , hr , Ny);
				
				indR   += 10;	
			}
			if(weaklearner == 0)		
			{
				sum    += (a*( (z*std) > th ) + b);	
			}
			else if(weaklearner == 1)
			{	
				sum    += ((2.0/(1.0 + exp(-2.0*epsi*(th*(z*std) + b)))) - 1.0);	
			}
			else if(weaklearner == 2)
			{
				sum    += a*sign((z*std) - th);
			}												
			indf      += 4;
		}
		
		sum_total     += sum;

		if((sum_total < thresh) && (cascade_type == 1))
		{
			yfx[0]    = -1.0;
			break;	
		}
		else if((sum < thresh) && (cascade_type == 0))	
		{
			yfx[0]    = -1.0;
			break;
		}
		indc      += 2; 
	}
	if(cascade_type == 1 )		
	{
		fx     = sum_total;	
	}
	else if (cascade_type == 0)
	{
		fx     = sum;	
	}
	/* Free pointers */

	free(Itemp);
	free(II);

	return fx;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int Round(double x)
{
	return (int)(x + 0.5);
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
	int h1 = h-1 , w1 = w-1 , x1 = x-1, y1 = y-1;

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
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
