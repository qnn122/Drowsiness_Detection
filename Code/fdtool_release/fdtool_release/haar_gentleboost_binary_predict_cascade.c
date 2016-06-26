
/*

  Predict data label with a Strong Classifier trained with haar_gentleboost_binary_model_cascade

  Usage
  ------

  [yest , fx] = haar_gentleboost_binary_predict_cascade(II , [options] );

  
  Inputs
  -------

  II                                    Integral of standardized Images  (Ny x Nx x N) in DOUBLE format
  options
         param                          Trained param structure (4 x T). param(: , i) = [featureIdx ; th ; a ; b]
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         Affine parameter(1 x T)
			                            b                         Bias parameter (1 x T)
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
        F                               Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle
        weaklearner                     Choice of the weak learner used in the training phase (default weaklearner = 0)
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(th,a)) = sigmoid(x ; a,b) in R
		epsi                            Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1)
        cascade_type                    Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)
        cascade                         Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
									    If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))
									    cascade(2 , :) reprensent thresholds for each segment

  Outputs
  -------
  
  yest                                  Estimated labels (1 x N) in INT8 format
  fx                                    Additive models (1 x N)

  To compile
  ----------


  mex  -output haar_gentleboost_binary_predict_cascade.dll haar_gentleboost_binary_predict_cascade.c

  mex  -f mexopts_intel10.bat -output haar_gentleboost_binary_predict_cascade.dll haar_gentleboost_binary_predict_cascade.c


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

  index                     = randperm(length(y));
  
  N                         = 2000;
  vect                      = [1:N , 5001:5001+N-1];
  indextrain                = vect(randperm(length(vect)));
  indextest                 = (1:length(y));
  indextest(indextrain)     = [];

  ytrain                    = y(indextrain);
  ytest                     = y(indextest);
  options.param             = haar_gentleboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);
  [ytest_est , fxtest]      = haar_gentleboost_binary_predict_cascade(II(: , : , indextest) , options);

  indp                      = find(ytest == 1);
  indn                      = find(ytest ==-1);

  tp                        = sum(ytest_est(indp) == ytest(indp))/length(indp)
  fp                        = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn)
  perf                      = sum(ytest_est == ytest)/length(ytest)

  [tpp , fpp , threshold]   = basicroc(ytest , fxtest);

  figure(1)
  plot(fpp , tpp , 'linewidth' , 2)
  axis([-0.02 , 1.02 , -0.02 , 1.02])


  [dum , ind]              = sort(ytest , 'descend');
  figure(2)
  plot(fxtest(ind))


  figure(3)
  plot(abs(options.param(3 , :)) , 'linewidth' , 2)
  grid on
  xlabel('Weaklearner m')
  ylabel('|a_m|')


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
	double         epsi;
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
void  haar_gentleboost_binary_predict_cascade(double * , int , int , int , struct opts , char *, double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{	
    double *II ;
	const int *dimsII;		
	double  param_default[80]      = {37175.00000,-12.32023,-1.35253, 0.59971,72871.00000,16.89163, 1.24400,-0.56781,58198.00000,-2.94586,-1.08418, 0.49939,66741.00000,-25.86427,-1.10336, 0.68034,8270.00000,-0.61001,-1.05877, 0.31355,51642.00000, 0.46534, 0.79673,-0.52647,21857.00000,-1.64872,-0.74512, 0.37631,3737.00000,-0.13907, 0.82898,-0.61126,3421.00000,-4.14634, 0.81135,-0.66546,11362.00000, 0.02812,-0.76322, 0.20181,60638.00000,18.43076, 0.77244,-0.20382,70861.00000,-32.79616,-0.77071, 0.61045,76798.00000,-20.96328,-0.81251, 0.60637,71280.00000,42.71988, 0.86371,-0.15101,63261.00000, 3.96094, 0.71051,-0.22937,19396.00000, 5.07520, 0.62837,-0.25674,2047.00000,-0.92711,-0.67379, 0.25054,45879.00000, 1.08460, 0.62295,-0.25773,3750.00000,-0.51966, 0.84909,-0.71660,67064.00000,-24.13888,-0.61725, 0.39873};
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	int Tcascade = 0;	
	int *dimsyest;
	struct opts options;
    char *yest;
	double *fx;
	int i , Ny , Nx , N;
	mxArray *mxtemp;
	double *tmp;
	int tempint;
	
	options.nR           = 4;
    options.nF           = 0;
	options.weaklearner  = 0; 
	options.epsi         = 1.0;
	options.cascade_type = 0;
	options.Ncascade     = 1;
	options.T            = 20;

    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) ==3) && (!mxIsEmpty(prhs[0])) && (mxIsDouble(prhs[0])) )
	{	
		II          = (double *)mxGetData(prhs[0]);	
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
			options.param                 = (double *)mxMalloc(80*sizeof(double));	
			for(i = 0 ; i < 80 ; i++)
			{
				options.param[i]          = param_default[i];	
			}	
			options.T                     = 20;
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
			options.cascade                 = (double *)mxMalloc(2*sizeof(double));
			options.cascade[0]              = (double) options.T;
			options.cascade[1]              = 0.0;		
		}	
	}
	else
	{
		options.param                 = (double *)mxMalloc(80*sizeof(double));	
		for(i = 0 ; i < 80 ; i++)
		{
			options.param[i]          = param_default[i];	
		}	
		options.T                     = 20;

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
	
	haar_gentleboost_binary_predict_cascade(II , Ny , Nx , N , options , yest , fx  );
	
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

void  haar_gentleboost_binary_predict_cascade(double *II , int Ny , int Nx , int N , struct opts options , char *yest, double *fx)
{
	double *param = options.param , *rect_param = options.rect_param, *cascade = options.cascade;
	unsigned int *F = options.F;
	int nR = options.nR , nF = options.nF , Ncascade = options.Ncascade;
	int weaklearner = options.weaklearner , cascade_type = options.cascade_type;
	double epsi = options.epsi;
	int t , n , c  , Tc;	
	int indc , indm , featureIdx , NyNx = Ny*Nx , indNyNx;	
	double z;
	double  th , a , b , thresc;	
	double sum , sum_total;
	
	indNyNx   = 0;
	
	for(n = 0 ; n < N ; n++)
	{	
		sum_total = 0.0;	
		indc      = 0;
		indm      = 0;
			
		for (c = 0 ; c < Ncascade ; c++)
		{	
			Tc      = (int) cascade[0 + indc];	
			thresc  = cascade[1 + indc];
			sum     = 0.0;
			
			for(t = 0 ; t < Tc ; t++)
			{			
				featureIdx = ((int) param[0 + indm]) - 1;	
				z          = haar_feat(II + indNyNx , featureIdx , rect_param , F , Ny , nR , nF);	
				th         = param[1 + indm];
				a          = param[2 + indm];
				b          = param[3 + indm];
				if(weaklearner == 0) /* Decision Stump */
				{
					sum       += (a*( z > th ) + b);	
				}
				if(weaklearner == 1) /* Perceptron */
				{
					sum       += ((2.0/(1.0 + exp(-2.0*epsi*(a*z + b)))) - 1.0);	
				}
				
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
		if(cascade_type == 1)
		{	
			fx[n]       = sum_total;	
			yest[n]     = sign(sum_total);
		}
		else if(cascade_type == 0)
		{
			fx[n]       = sum;	
			yest[n]     = sign(sum);
		}
		indNyNx    += NyNx;
	}	
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */

double haar_feat(double *II , int featidx , double *rect_param , unsigned int *F , int Ny , int nR , int nF)
{
	int x , xr , y , yr , w , wr , h , hr , r  ,  R , indR , indF = featidx*6;
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
		return (II[(y+h1) + (x+w1)*Ny] - (II[y1 + (x+w1)*Ny] + II[(y+h1) + x1*Ny]) + II[y1 + x1*Ny]);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
