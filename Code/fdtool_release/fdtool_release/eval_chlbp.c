
/*

  Eval Circular Histograms of Local Binary Patterns on a set of images X with a trained model

  Usage
  ------

  fx         = eval_chlbp(X , model);

  
  Inputs
  -------

  I                                     Input image (Ny x Nx x N) in UINT8 format
  
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
             dimsItraining              Size of the train images used in the mblbp computation, i.e. (ny x nx )
			 N                          Number of sampling points (1 x nR) (default [N=8])
			 R                          Vector of Radius (1 x nR) (default [R=1]) ny>2max(R)+1 & nx>2max(R)+1
			 map                        Mapping of the chlbp (2^Nmax x nR) in double format
			 shiftbox                   Shifting box parameters shiftbox (2 x 2 x nR) where [by , bx ; deltay , deltax] x nR (default shiftbox = [ny, nx ; 0 , 0])
	         cascade_type               Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade
             cascade                    Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.

                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
										If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))			
										cascade(2 , :) reprensent thresholds for each segment
  Outputs
  -------
  
  fx                                    Output matrix (1 x V) of the last stage/Strong classifier (cascade_type = 0/1) 

  To compile
  ----------

  mex  -output eval_chlbp.dll eval_chlbp.c
  mex  -f mexopts_intel10.bat -output eval_chlbp.dll eval_chlbp.c

  Example 1    Viola-Jones database
  ---------

  load viola_24x24.mat
  load model_detector_chlbp_24x24.mat
  thresh           = 0;


  indp             = find(y == 1);
  indn             = find(y ==-1);

  fx               = eval_chlbp(X , model);
  yest             = int8(sign(fx - thresh));


  tp               = sum(yest(indp) == y(indp))/length(indp)
  fp               = 1 - sum(yest(indn) == y(indn))/length(indn)
  perf             = sum(yest == y)/length(y)
  [tpp1 , fpp1 ]   = basicroc(y , fx);


  model.cascade    = [20 , 30 , 50 ; 0 , 0 , 0]; 
  fx_cascade       = eval_chlbp(X , model );
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
  title('CHLBP')


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
#define round(f)   ((f>=0)?(int)(f + .5):(int)(f - .5))
#define sign(a)    ((a) >= (0) ? (1.0) : (-1.0))
 
#define M_PI 3.14159265358979323846
#define huge 1e300
#define tiny 1e-6
#define NOBIN -1
#define intmax 32767 

struct model
{
	int     weaklearner;
	double  epsi;
	double *param;
	int     T;
	double  *dimsItraining;
	int     ny;
	int     nx;
	double *N;
	int     nR;
	double *R;
	double *map;
	int    *bin;
	double *shiftbox;
	int     cascade_type;
	double *cascade;
	int    Ncascade;
};
/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void qs( double * , int , int  ); 
int	number_chlbp_features(int , int , int * , double * , int );
void eval_chlbp(unsigned char * , int , int , int , struct model , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    unsigned char *I;
	struct model detector;
    const int *dimsI ;
    int numdimsI , Tcascade = 0;
    double *fx;	
	mxArray *mxtemp;
    int i , j , Ny , Nx , nR , V = 1, maxN  , maxR , indmaxN , powN , powmaxN , tempint , bincurrent , currentbin;	
	double *tmp , *mapsorted;

	detector.weaklearner  = 0; 
	detector.epsi         = 0.1;
	detector.cascade_type = 0;
	detector.Ncascade     = 1;
	
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

    Ny                                    = dimsI[0];  
    Nx                                    = dimsI[1];
	if(numdimsI > 2)
	{
		V                                 = dimsI[2];
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
		
		mxtemp                             = mxGetField( prhs[1] , 0, "dimsItraining" );
		if(mxtemp != NULL)
		{
			detector.dimsItraining         =  mxGetPr(mxtemp);              
			detector.ny                    = (int)detector.dimsItraining[0];
			detector.nx                    = (int)detector.dimsItraining[1];
			
			if ((Ny != detector.ny ) || (Nx != detector.nx ))       
			{		
				mexErrMsgTxt("I must be  nyxnx");	
			}
		}
		
		mxtemp                             = mxGetField( prhs[1] , 0, "N" );
		if(mxtemp != NULL)
		{	
			detector.N                     = (double *) mxGetData(mxtemp);	
			detector.nR                    = mxGetN(mxtemp);
			maxN                           = (int)detector.N[0];
			powmaxN                        = (int)pow(2 , maxN);	
			for(i = 1 ; i < detector.nR ; i++)
			{		
				maxN                       = max(maxN , (int)detector.N[i]);	
			}	
		}
		else	
		{
			detector.N                     = (double *)mxMalloc(sizeof(double));		
			detector.N[0]                  = 8;
			detector.nR                    = 1;
			maxN                           = 8;
			powmaxN                        = 256;		
		}

		nR                                 = detector.nR;
		mxtemp                             = mxGetField( prhs[1] , 0, "R" );		
		if(mxtemp != NULL)
		{
			detector.R                     = (double *) mxGetData(mxtemp);	
			if(mxGetN(mxtemp) != detector.nR)
			{
				mexErrMsgTxt("R must be (1 x nR)");	
			}

			maxR                           = (int)detector.R[0];
			for(i = 1 ; i < nR ; i++)
			{
				maxR                       = max(maxR , (int)detector.R[i]);
			}
		}
		else	
		{
			detector.R                     = (double *)mxMalloc(sizeof(double));
			detector.R[0]                  = 1;
		}
		
		mxtemp                             = mxGetField( prhs[1] , 0, "map" );
		if(mxtemp != NULL)
		{
			if((mxGetM(mxtemp) != powmaxN) && (mxGetN(mxtemp) != detector.nR))
			{		
				mexErrMsgTxt("map must be (2^maxN x nR) in double format");	
			}
			detector.map                   = (double *) mxGetData(mxtemp);
			
			/* Determine unique values in map vector */
			
			mapsorted                      = (double *)mxMalloc(powmaxN*sizeof(double));		
			detector.bin                   = (int *) mxMalloc(nR*sizeof(int));
			
			indmaxN                        = 0;
			for(j = 0 ; j < nR ; j++)
			{
				powN = (int) (pow(2 , (int) detector.N[j]));
				for ( i = 0 ; i < powN ; i++ )
				{			
					mapsorted[i] = detector.map[i + indmaxN];	
				}
				
				qs( mapsorted , 0 , powN - 1 );
				
				bincurrent                 = 0;
				currentbin                 = (int)mapsorted[0];
				
				for (i = 1 ; i < powN ; i++)
				{
					if (currentbin != mapsorted[i])
					{			
						currentbin         = (int)mapsorted[i];	
						bincurrent++;
					}
				}
				bincurrent++;			
				detector.bin[j]            = bincurrent;
				indmaxN                   += powmaxN;						
			}
		}
		else
		{	
			detector.map                   = (double *)mxMalloc(powmaxN*sizeof(double));		
			detector.bin                   = (int *) mxMalloc(nR*sizeof(int));
			indmaxN                        = 0;
			for(j = 0 ; j < nR ; j++)
			{	
				powN = (int) (pow(2 , (int) detector.N[j]));
				for(i = indmaxN ; i < powN + indmaxN ; i++)
				{				
					detector.map[i]        = i;	
				}
				indmaxN                   += powmaxN;
				detector.bin[j]            = powN;	
			}
		}	
		
		mxtemp                             = mxGetField( prhs[1] , 0, "shiftbox" );
		if(mxtemp != NULL)
		{	
			if((mxGetM(mxtemp) != 2) && (mxGetN(mxtemp) != 2) )
			{		
				mexErrMsgTxt("shiftbox must be (2 x 2 x nR) in double format");	
			}
			
			detector.shiftbox               = (double *) mxGetData(mxtemp);
			for(i = 0 ; i < nR ; i++)
			{
				if( (detector.shiftbox[0 + i*4] < 2.0*maxR+1) || (detector.shiftbox[2 + i*4] < 2.0*maxR+1) )
				{		
					mexErrMsgTxt("by and bx of shftbox must be > 2*max(R) + 1");	
				}
			}   			
		}
		else
		{	
			detector.shiftbox               = (double *)mxMalloc(4*sizeof(double));	
			detector.shiftbox[0]            = Ny;
			detector.shiftbox[1]            = 0.0;
			detector.shiftbox[2]            = Nx;
			detector.shiftbox[3]            = 0.0;			
		}

		mxtemp                              = mxGetField( prhs[1] , 0, "cascade_type" );
		if(mxtemp != NULL)
		{
			tmp                             = mxGetPr(mxtemp);	
			tempint                         = (int) tmp[0];
			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("cascade_type = {0,1}, force to 0");				
				detector.cascade_type       = 0;	
			}
			else
			{
				detector.cascade_type       = tempint;	
			}			
		}

		mxtemp                              = mxGetField( prhs[1] , 0, "cascade" );
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
    }
	else	
	{		
        mexErrMsgTxt("A detector structure is required");
	}
    
    /*------------------------ Output ----------------------------*/

    plhs[0]                                = mxCreateDoubleMatrix(1 , V , mxREAL);  
    fx                                     = mxGetPr(plhs[0]);

    /*------------------------ Main Call ----------------------------*/
	
	eval_chlbp(I , Ny , Ny , V , detector , fx);

	/*--------------------------- Free memory -----------------------*/
	
	mxFree(detector.bin);
	mxFree(mapsorted);

	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "N" )) == NULL )    
		{
			mxFree(detector.N);
		}
		if ( (mxGetField( prhs[1] , 0 , "R" )) == NULL )
		{
			mxFree(detector.R);
		}
		if ( (mxGetField( prhs[1] , 0 , "map" )) == NULL ) 
		{
			mxFree(detector.map);
		}
		if ( (mxGetField( prhs[1] , 0 , "shiftbox" )) == NULL)   
		{
			mxFree(detector.shiftbox);
		}
		if ( (mxGetField( prhs[1] , 0 , "cascade" )) == NULL)   
		{
			mxFree(detector.cascade);
		}
	}
	else
	{
		mxFree(detector.cascade);
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void eval_chlbp(unsigned char *I , int Ny , int Nx , int V , struct model detector  ,  double *f)
{
	double   *param = detector.param , *N = detector.N, *R = detector.R , *shiftbox = detector.shiftbox , *map = detector.map , *cascade = detector.cascade;
	double epsi = detector.epsi;
	int *bin = detector.bin;
	int nR = detector.nR , ny = detector.ny , nx = detector.nx , weaklearner = detector.weaklearner , cascade_type = detector.cascade_type  , Ncascade = detector.Ncascade , Tc;
	unsigned int *H , z;
	int nH , by , deltay , bx , deltax , bymax=-1 , bxmax=-1;
	double th , a , b  , radius , minx=huge , miny=huge , maxx=-huge , maxy=-huge , temp , thresc , sum_total , sum;
	int bsizey , bsizex , dy , dx , floory , floorx , origy , origx , minR=intmax , nynx = ny*nx , indnynx;
	int ly , lx , offsety , offsetx , indCx , indrx , indfx , indcx , dimDy , dimDx , dimD , indD , indbox;
	int nbin , bincurrent , maxN = -1 , Ncurrent , indmaxN , indnH , powmaxN , indc , indi;
	double *spoints ;
	int *D , *vectpow2;
	int i , j , l , m , n , r  , c , v, co , k , idxF;
	double x , y , tx  , ty ;
	double w1 , w2 , w3 , w4;
	int rx , ry , fx , cx  , fy , cy;

	nH                 = number_chlbp_features(ny , nx , bin , shiftbox , nR );
	H                  = (unsigned int *)malloc(nH*sizeof(double));

	for (r = 0 ; r < nR ; r++)
	{
		minR             = min(minR , 2*(int)R[r]);
		maxN             = max(maxN , (int) N[r]);
		bymax            = max(bymax , (int) shiftbox[0 + 4*r]);
		bxmax            = max(bxmax , (int) shiftbox[2 + 4*r]);
	}

	spoints              = (double *) malloc(2*maxN*sizeof(double));
	vectpow2             = (int *) malloc(maxN*sizeof(int));

	vectpow2[0]          = 1;
	for (i = 1 ; i < maxN ; i++)
	{
		vectpow2[i]      = vectpow2[i - 1]*2;
	}

	dimDy                = (bymax - minR);
	dimDx                = (bxmax - minR);
	dimD                 = dimDy*dimDx;
	D                    = (int *) malloc(dimD*sizeof(int)); /* (dy+1 x dx+1) */

	powmaxN              = (int) pow(2 , maxN);

	indnynx              = 0;
	indnH                = 0;

	for (v = 0 ; v < V ; v++)
	{
		indbox     = 0;
		indmaxN    = 0;
		co         = 0;

		for(i = 0 ; i < nH ; i++)
		{
			H[i]   = 0;
		}
		for (r = 0 ; r < nR ; r++)
		{
			by         = (int) shiftbox[0 + indbox];
			deltay     = (int) shiftbox[1 + indbox];
			bx         = (int) shiftbox[2 + indbox];
			deltax     = (int) shiftbox[3 + indbox];

			ly         = max(1 , (int) (floor(((ny - by)/(double) deltay))) + 1);
			offsety    = max(0 , (int)( floor(ny - ( (ly-1)*deltay + by + 1)) ));
			lx         = max(1 , (int) (floor(((nx - bx)/(double) deltax))) + 1);
			offsetx    = max(0 , (int)( floor(nx - ( (lx-1)*deltax + bx + 1)) ));

			dimDy      = (by - minR);
			dimDx      = (bx - minR);
			dimD       = dimDy*dimDx;
			radius     = R[r];
			Ncurrent   = (int) N[r];
			bincurrent = bin[r];
			a          = 2*M_PI/(double) Ncurrent;
			for (i = 0 ; i < Ncurrent ; i++)
			{
				temp                = -radius*sin(i*a);
				miny                = min(miny , temp);
				maxy                = max(maxy , temp);
				spoints[i]          = temp;
				temp                = radius*cos(i*a);
				minx                = min(minx , temp);
				maxx                = max(maxx , temp);
				spoints[i+Ncurrent] = temp;
			}
			floory           = (int) (floor(min(miny , 0)));
			floorx           = (int) (floor(min(minx , 0)));
			bsizey           = (int) (ceil(max(maxy , 0))) - floory + 1;
			bsizex           = (int) (ceil(max(maxx , 0))) - floorx + 1;
			dy               = by - bsizey;
			dx               = bx - bsizex;

			for(l = 0 ; l < lx ; l++) /* Loop shift on x-axis */
			{
				origx  = offsetx + l*deltax - floorx ;
				for(m = 0 ; m < ly ; m++)   /* Loop shift on y-axis  */
				{
					origy  = offsety + m*deltay - floory ;
					for(i = 0 ; i < dimD ; i++)
					{
						D[i] = 0;
					}
					for (n = 0 ; n < Ncurrent ; n++)  /* Loop over Ncurrent sampling points on circle of radius R */
					{
						nbin  = vectpow2[n];
						y     = spoints[n]            + origy;
						x     = spoints[n + Ncurrent] + origx;
						ry    = (int)round(y);
						rx    = (int)round(x);
						if( (fabs(x - rx) < tiny) && (fabs(y - ry) < tiny) )  /*  Linear interpolation */
						{
							indD      = 0;
							indrx     = rx*ny + indnynx;
							indCx     = origx*ny + indnynx;
							for(i = 0 ; i <= dx ; i++)
							{
								for(j = 0 ; j <= dy ; j++)
								{
									if( I[j + ry +  indrx] >= I[j + origy + indCx])
									{
										D[j + indD] += nbin;
									}
								}
								indD  += dimDy;
								indrx += ny;
								indCx += ny;
							}
						}
						else /*  Bilinear interpolation */
						{
							fy        = (int)floor(y);
							cy        = fy + 1;
							ty        = y - fy;

							fx        = (int)floor(x);
							cx        = fx + 1;
							tx        = x - fx;

							w1        = (1.0 - tx) * (1.0 - ty);
							w2        =        tx  * (1.0 - ty);
							w3        = (1.0 - tx) *      ty ;
							w4        =        tx  *      ty ;

							indD      = 0;
							indfx     = fx*ny + indnynx;
							indCx     = origx*ny + indnynx;

							for(i = 0 ; i <= dx ; i++)
							{
								indcx = indfx + ny ;

								for(j = 0 ; j <= dy ; j++)
								{
									temp = w1*I[j + fy + indfx] + w2*I[j + fy + indcx] + w3*I[j + cy + indfx] + w4*I[j + cy + indcx];
									if( temp >= I[j + origy + indCx])
									{								
										D[j + indD] += nbin;
									}
								}
								indD  += dimDy;
								indfx += ny;
								indCx += ny;
							}
						}
					}
					indD                = 0;
					for(i = 0 ; i <= dx ; i++)
					{
						for(j = indD ; j <= (dy + indD) ; j++)
						{
							k       = (int) map[D[j] + indmaxN];
							H[k + co]++;
						}
						indD            += dimDy;
					}
					co                  += bincurrent;
				}
			}
			indmaxN   += powmaxN;
			indbox    += 4;
		}
		indi          = 0;		
		indc          = 0;

		sum_total     = 0.0;
		for (c = 0 ; c < Ncascade ; c++)
		{
			Tc        = (int) cascade[0 + indc];
			thresc    = cascade[1 + indc];
			sum       = 0.0;
			for (i = 0 ; i < Tc ; i++)
			{
				idxF  = ((int) param[0 + indi] - 1);
				th    =  param[1 + indi];
				a     =  param[2 + indi];
				b     =  param[3 + indi];
				z     =  H[idxF];
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
				indi      += 4;
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
		if(cascade_type == 1 )
		{
			f[v]     = sum_total;
		}
		else if(cascade_type == 0 )
		{
			f[v]     = sum;
		}
		indnynx    += nynx;
	}
	free(spoints);
	free(vectpow2);
	free(D);
	free(H);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void qs(double  *a , int lo, int hi)
{
	int i=lo, j=hi;
	double x=a[(lo+hi)/2] , h;
	do
	{    
		while (a[i]<x) i++; 
		while (a[j]>x) j--;
		if (i<=j)
		{
			h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			i++; 
			j--;
		}
	}
	while (i<=j);

	if (lo<j) qs(a , lo , j);
	if (i<hi) qs(a , i , hi);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
int	number_chlbp_features(int ny , int nx  , int *bin , double *shiftbox , int nR)
{
	int l , ind = 0 , nH = 0 , sy , sx;
	for (l = 0 ; l < nR ; l++)
	{
		sy          = max(1 , (int) (floor( ((ny - (int) shiftbox[0 + ind])/shiftbox[1 + ind]) )) + 1);       
		sx          = max(1 , (int) (floor(((nx - (int) shiftbox[2 + ind])/shiftbox[3 + ind]))) + 1);
		nH         += bin[l]*sy*sx;
		ind        += 4;
	}
	return nH;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
