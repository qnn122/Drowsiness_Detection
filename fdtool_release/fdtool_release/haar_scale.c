
/*

  Haar's features of image I standardized and scaled wrt the trained database

  Usage
  ------

  z                                     = haar_scale(I , model );

  
  Inputs
  -------

  I                                     Images (Ny x Nx x N) in UINT8 format 
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
			 dimsItraining              Size of the train images used in the haar computation, i.e. (ny x nx )
             rect_param                 Features rectangles parameters (10 x nR), where nR is the total number of rectangles for the patterns.
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
             F                          Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle
             cascade_type               Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade
   

  Outputs
  -------
  
  z                                    Haar features matrix (nF x P) in INT16 format for each positions (y,x) in [1+h,...,ny-h]x[1+w,...,nx-w] and (w,h) integral block size.                              

  To compile
  ----------

  mex  -g -output haar_scale.dll haar_scale.c

  mex  -f mexopts_intel10.bat -output haar_scale.dll haar_scale.c


  Example 1
  ---------


  load model_detector_haar_24x24.mat
  load viola_24x24

  Nimage           = 110;
  cascade          = [30 ; 0];

  cumfx1           = zeros(1 , 100);
  cumfx2           = zeros(1 , 100);


  Origx            = 100;
  Origy            = 60;
  size_sub         = 55;


  Origx            = 145;
  Origy            = 225;
  size_sub         = 45;

  Origx            = 477;
  Origy            = 340;
  size_sub         = 55;


  Origx            = 1160;
  Origy            = 410;
  size_sub         = 55;

  Origx            = 960;
  Origy            = 365;
  size_sub         = 55;




  I                = (rgb2gray(imread('class57.jpg')));

  im1              = (I(Origy:Origy + size_sub - 1 , Origx:Origx + size_sub - 1));
  im2              = imresize(im1 , [24 , 24]);
  im3              = (X(: , : , Nimage));
  im4              = (X(: , : , Nimage + 5000));
  
  z1               = haar_scale(im1 , model);
  z2               = haar_scale(im2 , model);
  z3               = haar(im3, model.rect_param , model.F);
  z4               = haar(im4 , model.rect_param , model.F);


  eval_haar_subwindow(im1 , model , cascade)
  eval_haar(im2 , model , cascade)


  for i = 1:100
   
	 cumfx1(i) = eval_haar_subwindow(im1 , model , [i ; 0]);
	 cumfx2(i) = eval_haar(im2 , model , [i ; 0]);

  end


  corrcoef(z1,z2)
  
  figure(1)
  imagesc(I)
  hold on
  rectangle('Position' , [Origx , Origy , size_sub , size_sub] ,'Edgecolor',[0,1,0],'LineWidth',2);
  colormap(gray);
  hold off

  figure(2)
  imagesc(im1)
  colormap(gray)

  figure(3)
  imagesc(im2)
  colormap(gray)

  figure(4)
  plot(1:length(z1) , z1 , 1:length(z2) , z2 , 'r' , 1:length(z3) , z3 , 'g' , 1:length(z4) , z4 , 'm')
  legend('Scaled to train database' , 'Pre-interpolate' , 'Face VJ database' , 'Non-Face VJ database')


  figure(5)

  plot(1:100 , cumfx1 , 1:100 , cumfx2 , 'r')
  legend('Scaling' , 'Interpolating')




  Example 2
  ---------

  load viola_24x24
  load model_detector_haar_24x24.mat

  Nimage           = 110;

  im1              = double(X( : , : , Nimage));
  im2              = imresize(im1 , [36 , 36]);
  
  z1               = haar_scale(im1 , model);
  z2               = haar_scale(im2 , model);
  z3               = haar(im1 , model.rect_param , model.F);

  corrcoef(z1,z2)

  figure(1)
  imagesc(im1)
  colormap(gray)

  figure(2)
  imagesc(im2)
  colormap(gray)

  figure(3)
  plot(1:length(z1) , z1 , 1:length(z2) , z2 , 'r')
  legend('Original database' , 'Scaled')




 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/20/2009

 Reference ""


*/

#include <math.h>
#include <mex.h>
struct model
{
	double  *dimsItraining;
	int     ny;
	int     nx;
	double  *rect_param;
	int     nR;
	unsigned int  *F;
	int     nF;
};

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

int Round(double);
int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
void haar_scale(unsigned char * , int , int  , int , struct model  , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{    
	unsigned char *I;
	const int *dimsI;
	int numdimsI;
	struct model detector;
	mxArray *mxtemp;
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
    int i , Ny , Nx , N = 1;
	double *z;
	int numdimsz = 2;
	int *dimsz;
	
	detector.ny             = 24;
	detector.nx             = 24;
    detector.nR             = 4;
  
    /* Input 1  */
	    
    if ((nrhs > 0) && !mxIsEmpty(prhs[0]) && mxIsUint8(prhs[0]))    
    {        
		dimsI    = mxGetDimensions(prhs[0]);
        numdimsI = mxGetNumberOfDimensions(prhs[0]);

		
		I        = (unsigned char *) mxGetData(prhs[0]);
		Ny       = dimsI[0];	
		Nx       = dimsI[1];

		if(numdimsI > 2)
		{
			N    = dimsI[2];
		}
    }
	else
	{
		mexErrMsgTxt("I must be (Ny x Nx x N) in DOUBLE format");
	}
    
    /* Input 2  */
    
    if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )   
    {					
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
    }
	else	
	{	
		detector.rect_param            = (double *)mxMalloc(40*sizeof(double));
		
		for(i = 0 ; i < 40 ; i++)
		{		
			detector.rect_param[i]     = rect_param_default[i];	
		}	

		detector.nF                    = number_haar_features(detector.ny , detector.nx , detector.rect_param , detector.nR);	
		detector.F                     = (unsigned int *)mxMalloc(6*detector.nF*sizeof(int));
		haar_featlist(detector.ny , detector.nx , detector.rect_param , detector.nR , detector.F);
	}

    /*----------------------- Outputs -------------------------------*/

    /* Output 1  */

	dimsz         = (int *)mxMalloc(2*sizeof(int));
	dimsz[0]      = detector.nF;
	dimsz[1]      = N;
	plhs[0]       = mxCreateNumericArray(numdimsz , dimsz , mxDOUBLE_CLASS , mxREAL);

	z             = mxGetPr(plhs[0]);
	
    /*------------------------ Main Call ----------------------------*/
		
	haar_scale(I , Ny , Nx , N , detector ,  z);

	/*----------------- Free Memory --------------------------------*/
	
	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "rect_param" )) == NULL )
		{
			mxFree(detector.rect_param);
		}
		if ( (mxGetField( prhs[1] , 0 , "F" )) == NULL )
		{
			mxFree(detector.F);
		}
	}
	else
	{
		mxFree(detector.rect_param);
		mxFree(detector.F);
	}

	mxFree(dimsz);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void haar_scale(unsigned char *I , int Ny , int Nx , int P , struct model detector , double *z)
{
	int  p , indF , NxNy = Nx*Ny , indNxNy = 0 ;
	double *rect_param = detector.rect_param;
	unsigned int *F = detector.F;
	int nx = detector.nx , ny = detector.ny , nF = detector.nF;
	unsigned int f , indnF = 0;
	int x , xr , y , yr  , w , wr  , h , hr , i , r , R , indR ;
	int coeffw , coeffh;
	int last = NxNy - 1;
	double val , s , var , mean , std , cteNxNy = 1.0/NxNy , scalex = (Nx - 1 )/(double)nx , scaley = (Ny - 1 )/(double)ny ;
	double ctescale = 1.0/(scalex*scaley);
	unsigned int *II  , *Itemp , tempI;

	II          = (unsigned int *)malloc(NxNy*sizeof(unsigned int));
	Itemp       = (unsigned int *)malloc(NxNy*sizeof(unsigned int));
				
	for(p = 0 ; p < P ; p++)
	{	
		MakeIntegralImage((I + indNxNy) , II , Nx , Ny , Itemp);
		var       = 0.0;
		for(i = 0 ; i < NxNy ; i++)
		{
			tempI      = I[i + indNxNy];	
			var       += (tempI*tempI);
		}
				
		var      *= cteNxNy;
		mean      = II[last]*cteNxNy;
		std       = ctescale/sqrt(var - mean*mean);
		
		indF      = 0;
		
		for (f = 0 ; f < nF ; f++)
		{		
			x     = F[1 + indF];	
			y     = F[2 + indF];
			w     = F[3 + indF];
			h     = F[4 + indF];
			indR  = F[5 + indF];
			R     = (int) rect_param[3 + indR];
			
			val   = 0.0;
			
			for (r = 0 ; r < R ; r++)
			{
				coeffw  = w/(int)rect_param[1 + indR];	
				coeffh  = h/(int)rect_param[2 + indR];
				xr      = Round(scalex*(x + (coeffw*(int)rect_param[5 + indR])));
				yr      = Round(scaley*(y + (coeffh*(int)rect_param[6 + indR])));
				wr      = Round(scalex*coeffw*(int)(rect_param[7 + indR]));
				hr      = Round(scaley*coeffh*(int)(rect_param[8 + indR]));
				s       = rect_param[9 + indR];
				val    += s*Area(II , xr  , yr  , wr , hr , Ny);
				indR   += 10;
			}
			z[f + indnF]    = val*std;		
			indF           += 6;
		}
		indNxNy   += NxNy;
		indnF     += nF;
	}
			
	free(II);
	free(Itemp);		
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
unsigned int Area(unsigned int *II , int x , int y , int w , int h , int Ny)
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
/*--------------------------------------------------------------------------------------------------------------------------------------------- */
int Round(double x)
{
	return (int)(x + 0.5);
}
/*----------------------------------------------------------------------------------------------------------------------------------------------*/
