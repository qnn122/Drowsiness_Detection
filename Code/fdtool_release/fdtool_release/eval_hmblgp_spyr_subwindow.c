/*

  Evaluate model on image I of fast Histogram of MBLGP features through Histogram Integral and trained by Linear SVM  

  Usage
  ------

  [fx , y , H , IIR , R]               = eval_hmblgp_spyr_subwindow(I , [model]);

  
  Inputs
  -------

  I                                     Input image (Ny x Nx) in UINT8 format
  
  model                                 Trained model structure

             w                          Trained model with a Linear SVM, weight vector (1 x ((1+improvedLGP)*Nbins*nH*nscale+addbias)) 
			                            where Nbins = ([256,59,36,10]*(improvedLGP+1)) if cs_opt = 0, Nbins = ([16,15,10,10]*(improvedLGP+1)) if cs_opt = 1
			 addbias                    Add bias or not for prediction (1/0) 
             homtable                   Precomputed table for homogeneous additive Kernel approximation (used when model.n > 0)
			 n                          Order approximation for the homogeneous additive Kernel
			 L                          Sampling step (default L = 0.5);
			 kerneltype                 0 for intersection kernel, 1 for Jensen-shannon kernel, 2 for Chi2 kernel (default kerneltype = 0)
		     numsubdiv                  Number of subdivisions (default numsubdiv = 8);
             minexponent                Minimum exponent value (default minexponent = -20)
             maxexponent                Maximum exponent value (default minexponent = 8)
             spyr                       Spatial Pyramid (nspyr x 5) (default [1 , 1 , 1 , 1 , 1] with nspyr = 1)
                                        where spyr(i,1) is the ratio of Ny in y axis of the blocks at level i (by(i) = spyr(i,1)*Ny)
                                        where spyr(i,2) is the ratio of nx in x axis of the blocks at level i (bx(i) = spyr(i,3)*nx)
                                        where spyr(i,3) is the ratio of Ny in y axis of the shifting at level i (deltay(i) = spyr(i,2)*Ny)
                                        where spyr(i,4) is the ratio of nx in x axis of the shifting at level i (deltax(i) = spyr(i,4)*nx)
                                        where spyr(i,5) is the weight's histogram associated to current level pyramid (w(i) = spyr(i,1)*spyr(i,2))
										total number of subwindows nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1))
			 nH                         Number of subwindows associated with spyr (default nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1)))
             scale                      Multi-Scale vector (1 x nscale) (default scale = 1) where scale(i) = s is the size's factor to apply to each 9 blocks
                                        in the LGP computation, i = 1,...,nscale
			 cs_opt                     Center-Symetric LGP : 1 for computing CS-MBLBP features, 0 : for MBLBP (default cs_opt = 0)
             improvedLGP                0 for default 8 bits encoding, 1 for 9 bits encoding (8 + central area)
 	         norm                       Normalization vector (1 x 3) : [for all subwindows, for each subwindows of a pyramid level, for each subwindows]
                                        norm = 0 <=> no normalization, norm = 1 <=> v=v/(sum(v)+epsi), norm = 2 <=> v=v/sqrt(sum(v²)+epsi²),
	                                    norm = 3 <=> v=sqrt(v/(sum(v)+epsi)) , norm = 4 <=> L2-clamped (default norm = [0 , 0 , 4])
			 clamp                      Clamping value (default clamp = 0.2)
	         maptable                   Mapping table for LGP codes. LGP code belongs to {0,...,b}, where b is defined according to following table
								        |maptable | cs_opt = 0, improvedLGP = 0 | cs_opt = 0, improvedLGP = 1 | cs_opt = 1, improvedLGP = 0 | cs_opt = 1, improvedLGP = 1|
										|   0     |           255               |              511            |            15               |              31            |
										|   1     |           58                |              117            |            14               |              29            |
										|   2     |           35                |              71             |            5                |              11            |
										|   3     |           9                 |              19             |            5                |              11            |
             rmextremebins              Force to zero bin = {0 , b} if  rmextremebins = 1 where b is defined in previous tab (default rmextremebins = 1)

 
  Outputs
  -------
  
  fx                                    Predicted value for image I 
  y                                     Predicted label, i.e. y = sign(fx)
  H                                     Histogram of MBLBP computed for image I through fast Histogram Integral ((1+improvedLGP)*Nbins*nH*nscale) x 1)
  IIR                                   Integral Images for each bin and scale (Ny x Nx x(1+improvedLGP)*Nbins*nscale) in UINT32 format
  R                                     MBLBP maps per bin value (Ny x Nx x(1+improvedLGP)*Nbins*nscale) in UINT8 format

  To compile
  ----------

  mex  -g eval_hmblgp_spyr_subwindow.c

  mex  eval_hmblgp_spyr_subwindow.c

  mex  -f mexopts_intel10.bat eval_hmblgp_spyr_subwindow.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat eval_hmblgp_spyr_subwindow.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"

  or with the matfx option

  mex  -Dmatfx -f mexopts_intel10.bat eval_hmblgp_spyr_subwindow.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -Dmatfx -f mexopts_intel10.bat eval_hmblgp_spyr_subwindow.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"


  Example 1
  ---------

  clear,close all
  load                     viola_24x24.mat
  I                        = X(: , : , 100);

 % model                    = load('modelw.mat');
%  model.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/2 , 1/2 , 1/4 , 1/4 , 1/4];
  model.spyr               = [1/2 , 1 , 1/2 , 1 , 1];
  model.nH                 = sum(floor(((1 - model.spyr(:,1))./(model.spyr(:,3)) + 1)).*floor((1 - model.spyr(:,2))./(model.spyr(:,4)) + 1));
  model.scale              = [1];
  model.maptable           = 0;
  model.cs_opt             = 0;
  model.improvedLGP        = 0;
  model.rmextremebins      = 0;
  model.norm               = [0 , 0 , 0];
  model.clamp              = 0.2;
  model.n                  = 0;
  model.L                  = 1;
  model.kerneltype         = 0;
  model.numsubdiv          = 8;
  model.minexponent        = -20;
  model.maxexponent        = 8;
  model.addbias            = 1;

 if(model.cs_opt == 0)
 if(model.maptable == 0)
     model.Nbins           = 256;
 elseif(model.maptable == 1)
     model.Nbins           = 59;
 elseif(model.maptable == 2)
     model.Nbins           = 36;
 elseif(model.maptable == 3)
     model.Nbins           = 10;
 end
 else
 if(model.maptable == 0)
     model.Nbins           = 16;
 elseif(model.maptable == 1)
     model.Nbins           = 15;
 elseif(model.maptable == 2)
     model.Nbins           = 6;
 elseif(model.maptable == 3)
     model.Nbins           = 6;
 end
 end

  model.w                   = randn(1,model.Nbins*length(model.scale)*model.nH+model.addbias);

  vect                      = reshape(1:model.Nbins , [1 , 1 , model.Nbins]);
  [fx , yfx , H , IIR , R]  = eval_hmblgp_spyr_subwindow(I , model);
  LGP                       = sum(double(R).*vect(ones(size(I,1),1),ones(1,size(I,2)),:) , 3);
  H1                        = mlhmslgp_spyr(I , model );

  H2                        = zeros(model.Nbins*(1+model.improvedLGP)*model.nH*length(model.scale) , 1);

  offset                    = 1;

  for i = 1:256
    H2(i)                       = aire(R(: , : , i) , 1+0*offset  , 1+0*offset , 12-1*offset , 24-1*offset);
	H2(i+256)                   = aire(R(: , : , i) , 13+1*offset , 1+0*offset , 12-1*offset , 24-1*offset); %2
  end

[H, H1 , H2]

plot(1:512,H,1:512,H1 , 'r' , 1:512 , H2 , 'k')

[H(1:256)+H(257:512) , H1(1:256)+H1(257:512) , squeeze(sum(sum(R,1))) ]

[H(1:256) , H1(1:256) , H2(1:256)]

[H(257:512) ,H1(257:512) , H2(257:512) ]


  Example 2
  ---------

  clear,close all

  load Itest

  model                    = load('modelw8.mat');
%  model.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/2 , 1/2 , 1/4 , 1/4 , 1/4];
%  model.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/2 , 1/2 , 1/2 , 1/2 , 1/4];
  model.spyr                = [1 , 1 , 1 , 1 , 1 ;  1/4 , 1/4 , 1/4 , 1/4 , 1/16];
  model.nH                 = sum(floor(((1 - model.spyr(:,1))./(model.spyr(:,3)) + 1)).*floor((1 - model.spyr(:,2))./(model.spyr(:,4)) + 1));
  model.scale              = 1; %;[1 , 2];
  model.maptable           = 1;
  model.improvedLGP        = 0;
  model.rmextremebins      = 1;
  model.norm               = [0 , 0 , 4];
  model.clamp              = 0.2;
  model.n                  = 0;
  model.L                  = 1;
  model.kerneltype         = 0;
  model.numsubdiv          = 8;
  model.minexponent        = -20;
  model.maxexponent        = 8;
  model.addbias            = 1;


  figure(1)
  imagesc(I)
  colormap(gray);

  Icrop                    = imcrop(I);
  [Ny , nx]                = size(Icrop);

  [fx, yfx , H ]           = eval_hmblgp_spyr_subwindow(Icrop , model);

  figure(2)

  imagesc(Icrop)
  colormap(gray);


  title(sprintf('Direct Crop image, fx = %4.3f' , fx));
  drawnow

  Icrop_interp              = imresize(Icrop , [128 , 128]);

  [fx1 , yfx1 , H1]         = eval_hmblgp_spyr_subwindow(Icrop_interp , model);

  figure(3)

  imagesc(Icrop_interp)
  colormap(gray);
  title(sprintf('Resized image, fx = %4.3f' , fx1));
  drawnow

  figure(4)
  plot(H)
  title(sprintf('Direct Crop image, fx = %4.3f' , fx));
  drawnow

  figure(5)
  plot(H1)
  title(sprintf('Resized image, fx = %4.3f' , fx1));
  drawnow

  vect  = (1:-0.01:0.7);
  lvect = length(vect);
  fx1   = zeros(1 , lvect);
  fx2   = zeros(1 , lvect);
  

  co    =  1;
  for i = vect
     Itemp                  = imresize(Icrop , [Ny , nx].^i);
	 H1(: , co)             = mlhmslbp_spyr(Itemp , model );
	 fx1(co)                = model.w(1:end-1)*H1(: , co) + model.w(end);
	 yfx1                   = sign(fx1(co));
     [fx2(co) , yfx2 , H2]  = eval_hmblgp_spyr_subwindow(Itemp , model);
     co                     = co+1;
  end

  figure(6)
  plot(vect , fx1 , vect , fx2 , 'r')



  Example 2
  ---------

  clear,close all

  load Itest
  load model_hmblgp_R4



  figure(1)
  imagesc(I)
  colormap(gray);

  Icrop                    = imcrop(I);
  [Ny , nx]                = size(Icrop);

  [fx, yfx , H ]           = eval_hmblgp_spyr_subwindow(Icrop , model);

  figure(2)

  imagesc(Icrop)
  colormap(gray);


  title(sprintf('Direct Crop image, fx = %4.3f' , fx));
  drawnow

  Icrop_interp       = imresize(Icrop , [128 , 128]);

  [fx1 , yfx1 , H1]  = eval_hmblgp_spyr_subwindow(Icrop_interp , model);

  figure(3)

  imagesc(Icrop_interp)
  colormap(gray);
  title(sprintf('Resized image, fx = %4.3f' , fx1));
  drawnow

  figure(4)
  plot(H)
  title(sprintf('Direct Crop image, fx = %4.3f' , fx));
  drawnow

  figure(5)
  plot(H1)
  title(sprintf('Resized image, fx = %4.3f' , fx1));
  drawnow


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 02/26/2011

 Reference ""


*/


#include <math.h>
#include <mex.h>

#ifdef OMP 
 #include <omp.h>
#endif

#define tiny 1e-8
#define verytiny 1e-15
#define PI 3.14159265358979323846


#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif
#define sign(a)    ((a) >= (0) ? (1) : (-1))
 
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

struct model
{
	double         *w;
	int             nw;
	int             addbias;
	int             n;
	double          L;
	int             kerneltype;
	int             numsubdiv;
	int             minexponent;
	int             maxexponent;
	double         *homtable;
	int             nhomtable;
	double         *scale;
	int             nscale;
	double         *spyr;
	int             nspyr;
	int             nH;
	int             cs_opt;
	int             improvedLGP;
	int             rmextremebins;
	double         *norm;
	double          clamp;
	int             maptable;
	int             Nbins;

#ifdef OMP 
    int            num_threads;
#endif

};

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */
int	number_histo_lbp(double * , int , int );
int Round(double );
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
void qsindex (double  *, int * , int , int );
void compute_mblgp(unsigned int * , unsigned int * , struct model , int , int , int , unsigned char * );
int eval_hmblgp_spyr_subwindow(unsigned int * , double * , int , int , int , int , double  , struct model   , double *);
int eval_hmblgp_spyr_subwindow_hom(unsigned int * , double * , int , int , int , int , double  , struct model   , double *);
void homkertable(struct model  , double * );
void eval_hmblgp_spyr(unsigned char * , int , int , struct model , double * , double *, double *, unsigned int * , unsigned char *);
/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *I;
	struct model detector;
	const int *dimsI;
	int dimsR[3];
	int numdimsI;
	double *y , *fx , *H;
	unsigned int *IIR;
	unsigned char *R;
	double norm_default[3] = {0 , 0 , 4};
	mxArray *mxtemp;	    
	int Ny , Nx , tempint , Nbins = 256 , i;
	double *tmp;
	double temp;

#ifdef matfx
	double *fxmat;
#endif
	detector.addbias        = 0;
    detector.n              = 0;
	detector.L              = 0.5;
	detector.kerneltype     = 0;
	detector.numsubdiv      = 8;;
	detector.minexponent    = -20;
	detector.maxexponent    = 8;
	detector.maptable       = 0;
	detector.nscale         = 1;
	detector.nspyr          = 1;
	detector.nH             = 1;
	detector.cs_opt         = 0;
	detector.improvedLGP    = 0;
	detector.rmextremebins  = 1;
	detector.clamp          = 0.2;

#ifdef OMP 
    detector.num_threads    = -1;
#endif

	if ((nrhs < 1))       
	{	
		mexErrMsgTxt("At least 1 input is requiered for detector_mblbp");	
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
		mxtemp                            = mxGetField( prhs[1] , 0, "addbias" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 1) )
			{
				mexPrintf("addbias = {0,1}, force to 0");					
				detector.addbias          = 0;	
			}
			else
			{
				detector.addbias          = tempint;	
			}			
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "n" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) )
			{
				mexPrintf("n >= 0, force to 0");					
				detector.n                = 0;	
			}
			else
			{
				detector.n                = tempint;	
			}			
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "L");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			temp                          = tmp[0];
			if( (temp < 0.0) )
			{
				mexPrintf("L >= 0, force to 0.5\n");	
				detector.L                = 0.5;
			}
			else
			{
				detector.L               = temp;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "kerneltype");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0) ||  (tempint > 2))
			{
				mexPrintf("kerneltype = {0,1,2}, force to 0\n");	
				detector.kerneltype        = 1;
			}
			else
			{
				detector.kerneltype        = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "numsubdiv");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 1) )
			{
				mexPrintf("numsubdiv > 0 , force to 8\n");	
				detector.numsubdiv        = 8;
			}
			else
			{
				detector.numsubdiv        = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "minexponent");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			detector.minexponent          = (int) tmp[0];
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "maxexponent");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < detector.minexponent) )
			{
				mexPrintf("maxexponent > minexponent , force to 8\n");	
				detector.maxexponent      = detector.minexponent + 2;
			}
			else
			{
				detector.maxexponent      = tempint;
			}
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "homtable" );
		if(mxtemp != NULL)
		{	
			detector.homtable             = mxGetData(mxtemp);	
			detector.nhomtable            = mxGetN(mxtemp);
			if(detector.nhomtable != ((2*detector.n+1)*(detector.maxexponent - detector.minexponent + 1)*detector.numsubdiv))
			{
				mexErrMsgTxt("homtable must be (1 x (2*n+1)*(maxexponent - minexponent + 1)*numsubdiv)");
			}
		}
		else
		{
			if(detector.n > 0)
			{
				detector.homtable         = (double *)mxMalloc( ((2*detector.n+1)*(detector.maxexponent - detector.minexponent + 1)*detector.numsubdiv)*sizeof(double)); 
                homkertable(detector , detector.homtable);
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "scale");
		if(mxtemp != NULL)
		{
			if( mxGetM(mxtemp) != 1 )
			{
				mexErrMsgTxt("scale must be (1 x nscale) in double format\n");
			}
			detector.scale                = mxGetPr(mxtemp);
			detector.nscale               = mxGetN(mxtemp);
		}
		else
		{
			detector.nscale               = 1;
			detector.scale                = (double *)mxMalloc(sizeof(double));
			detector.scale[0]             = 1.0;
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "spyr");
		if(mxtemp != NULL)
		{
			if( mxGetN(mxtemp) != 5 )
			{
				mexErrMsgTxt("spyr must be (nscale x 5) in double format\n");
			}
			detector.spyr                 = mxGetPr(mxtemp);
			detector.nspyr                = mxGetM(mxtemp);
		}
		else
		{
			detector.nspyr                 = 1;
			detector.spyr                  = (double *)mxMalloc(5*sizeof(double));
			detector.spyr[0]               = 1.0;
			detector.spyr[1]               = 1.0;
			detector.spyr[2]               = 1.0;
			detector.spyr[3]               = 1.0;
			detector.spyr[4]               = 1.0;
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "nH");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0))
			{
				mexPrintf("nH must be positive, force to 1\n");	
				detector.nH               = 1;
			}
			else
			{
				detector.nH               = tempint;
			}
		}
		else
		{
			detector.nH                  = number_histo_lbp(detector.spyr , detector.nspyr , detector.nscale);
		}

		mxtemp                           = mxGetField(prhs[1] , 0 , "cs_opt");
		if(mxtemp != NULL)
		{	
			tmp                           = mxGetPr(mxtemp);		
			tempint                       = (int) tmp[0];	
			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("cs_opt = {0,1}, force to 0");	
				detector.cs_opt           = 0;			
			}
			else
			{
				detector.cs_opt           = tempint;	
			}
			if(detector.cs_opt == 1)
			{
				Nbins                     = 16;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "improvedLGP");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("improvedLGP = {0,1}, force to 0\n");	
				detector.improvedLGP      = 0;
			}
			else
			{
				detector.improvedLGP      = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "rmextremebins");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("rmextremebins = {0,1}, force to 0\n");	
				detector.rmextremebins     = 0;
			}
			else
			{
				detector.rmextremebins     = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "norm");
		if(mxtemp != NULL)
		{
			if( mxGetN(mxtemp) != 3 )
			{
				mexErrMsgTxt("norm must be (1 x 3) in double format\n");
			}
			detector.norm                  = mxGetPr(mxtemp);
			for (i = 0 ; i < 3 ; i++)
			{
				if((detector.norm[i] < 0) || (detector.norm[i] > 4))
				{
					mexErrMsgTxt("norm must be (1 x 3) in double format\n");
				}
			}
		}
		else
		{
			detector.norm                 = (double *)mxMalloc(3*sizeof(double));
			for(i = 0 ; i < 3 ; i++)
			{
				detector.norm[i]          = norm_default[i];
			}	
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "clamp");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			temp                          = tmp[0];
			if( (temp < 0.0) )
			{
				mexPrintf("clamp must be >= 0, force to 0.2\n");	
				detector.clamp            = 0.2;
			}
			else
			{
				detector.clamp            = temp;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "maptable");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0) || (tempint > 3))
			{
				mexPrintf("maptable = {0,1,2,3}, force to 0\n");	
				detector.maptable          = 0;
			}
			else
			{
				detector.maptable          = tempint;
			}
		}

		mxtemp                             = mxGetField( prhs[1] , 0, "w" );
		if(mxtemp != NULL)
		{	
			detector.w                     = mxGetData(mxtemp);	
			detector.nw                    = mxGetN(mxtemp);	
		}
		else
		{
			detector.w                    = (double *)mxMalloc((1+detector.improvedLGP)*Nbins*detector.nH*detector.nscale*sizeof(double));
		}

#ifdef OMP 
		mxtemp                            = mxGetField( prhs[1] , 0, "num_threads" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < -2))
			{								
				detector.num_threads      = -1;
			}
			else
			{
				detector.num_threads      = tempint;	
			}			
		}
#endif
	}
	else	
	{	
		detector.nspyr                 = 1;
		detector.spyr                  = (double *)mxMalloc(5*sizeof(double));

		detector.spyr[0]               = 1.0;
		detector.spyr[1]               = 1.0;
		detector.spyr[2]               = 1.0;
		detector.spyr[3]               = 1.0;
		detector.spyr[4]               = 1.0;

		detector.nscale                = 1;
		detector.scale                 = (double *)mxMalloc(1*sizeof(double));
		detector.scale[0]              = 1.0;

		detector.w                     = (double *)mxMalloc((1+detector.improvedLGP)*Nbins*detector.nH*detector.nscale*sizeof(double));

		if(detector.n > 0)
		{
			detector.homtable          = (double *)mxMalloc( ((2*detector.n+1)*(detector.maxexponent - detector.minexponent + 1)*detector.numsubdiv)*sizeof(double)); 
			homkertable(detector , detector.homtable);
		}
		detector.norm                  = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.norm[i]           = norm_default[i];
		}	
	}

	if(detector.cs_opt == 1)
	{
		if(detector.maptable == 0)
		{
			detector.Nbins                       = 16;
		}
		else if(detector.maptable == 1)
		{
			detector.Nbins                       = 15;
		}
		else if(detector.maptable == 2)
		{
			detector.Nbins                       = 6;
		}
		else if(detector.maptable == 3)
		{
			detector.Nbins                       = 6;
		}
	}
	else
	{
		if(detector.maptable == 0)
		{
			detector.Nbins                       = 256;
		}
		else if(detector.maptable == 1)
		{
			detector.Nbins                       = 59;
		}
		else if(detector.maptable == 2)
		{
			detector.Nbins                       = 36;
		}
		else if(detector.maptable == 3)
		{
			detector.Nbins                       = 10;
		}
	}
	if(detector.improvedLGP == 1)
	{
		detector.Nbins                          *= 2;
	}


	plhs[0]                            = mxCreateDoubleMatrix(1 , 1 , mxREAL);
	fx                                 = mxGetPr(plhs[0]);

	plhs[1]                            = mxCreateDoubleMatrix(1 , 1 , mxREAL);
	y                                  = mxGetPr(plhs[1]);

	plhs[2]                            = mxCreateDoubleMatrix(detector.Nbins*detector.nH*detector.nscale , 1 , mxREAL);
	H                                  = mxGetPr(plhs[2]);

	dimsR[0]                           = Ny;
	dimsR[1]                           = Nx;
	dimsR[2]                           = detector.Nbins*detector.nscale;

	plhs[3]                            = mxCreateNumericArray(3 , dimsR , mxUINT32_CLASS , mxREAL);
	IIR                                = (unsigned int *)mxGetPr(plhs[3]);

	plhs[4]                            = mxCreateNumericArray(3 , dimsR , mxUINT8_CLASS , mxREAL);
	R                                  = (unsigned char *)mxGetPr(plhs[4]);


	/*------------------------ Main Call ----------------------------*/

	eval_hmblgp_spyr(I , Ny , Nx  , detector , y , fx , H , IIR , R);

  /*--------------------------------------------------------------------- */

	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "spyr" )) == NULL )
		{
			mxFree(detector.spyr);
		}
		if ( (mxGetField( prhs[1] , 0 , "scale" )) == NULL )
		{
			mxFree(detector.scale);
		}
		if ( (mxGetField( prhs[1] , 0 , "w" )) == NULL )
		{
			mxFree(detector.w);
		}
		if ( ((mxGetField( prhs[1] , 0 , "homtable" )) == NULL) && (detector.n > 0) )
		{
			mxFree(detector.homtable);
		}
		if ( (mxGetField( prhs[1] , 0 , "norm" )) == NULL )
		{
			mxFree(detector.norm);
		}
	}
	else
	{
		mxFree(detector.spyr);
		mxFree(detector.scale);
		mxFree(detector.w);
        mxFree(detector.norm);
		if(detector.n > 0)
		{
			mxFree(detector.homtable);
		}
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void eval_hmblgp_spyr(unsigned char *I , int Ny , int Nx  , struct model detector , double *yfx , double *fx , double *H , unsigned int *IIR , unsigned char *R)
{
	unsigned int *II , *Itemp;
	int nscale = detector.nscale  , nH = detector.nH;
	int Nbins = detector.Nbins , cs_opt = detector.cs_opt;
	int maptable = detector.maptable , improvedLGP = detector.improvedLGP , n = detector.n;
	int NyNx = Ny*Nx , Nbinsnscale , NbinsnscalenH , powN = 256;
#ifdef OMP 
    int num_threads = detector.num_threads;
#endif
	int i , l , m , v ;
	int yest;
	double maxfactor = 0.0;

	unsigned int table_normal_8[256] = {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 , 55 , 56 , 57 , 58 , 59 , 60 , 61 , 62 , 63 , 64 , 65 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 , 96 , 97 , 98 , 99 , 100 , 101 , 102 , 103 , 104 , 105 , 106 , 107 , 108 , 109 , 110 , 111 , 112 , 113 , 114 , 115 , 116 , 117 , 118 , 119 , 120 , 121 , 122 , 123 , 124 , 125 , 126 , 127 , 128 , 129 , 130 , 131 , 132 , 133 , 134 , 135 , 136 , 137 , 138 , 139 , 140 , 141 , 142 , 143 , 144 , 145 , 146 , 147 , 148 , 149 , 150 , 151 , 152 , 153 , 154 , 155 , 156 , 157 , 158 , 159 , 160 , 161 , 162 , 163 , 164 , 165 , 166 , 167 , 168 , 169 , 170 , 171 , 172 , 173 , 174 , 175 , 176 , 177 , 178 , 179 , 180 , 181 , 182 , 183 , 184 , 185 , 186 , 187 , 188 , 189 , 190 , 191 , 192 , 193 , 194 , 195 , 196 , 197 , 198 , 199 , 200 , 201 , 202 , 203 , 204 , 205 , 206 , 207 , 208 , 209 , 210 , 211 , 212 , 213 , 214 , 215 , 216 , 217 , 218 , 219 , 220 , 221 , 222 , 223 , 224 , 225 , 226 , 227 , 228 , 229 , 230 , 231 , 232 , 233 , 234 , 235 , 236 , 237 , 238 , 239 , 240 , 241 , 242 , 243 , 244 , 245 , 246 , 247 , 248 , 249 , 250 , 251 , 252 , 253 , 254 , 255};
	unsigned int table_u2_8[256]     = {0 , 1 , 2 , 3 , 4 , 58 , 5 , 6 , 7 , 58 , 58 , 58 , 8 , 58 , 9 , 10 , 11 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 12 , 58 , 58 , 58 , 13 , 58 , 14 , 15 , 16 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 17 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 18 , 58 , 58 , 58 , 19 , 58 , 20 , 21 , 22 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 23 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 24 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 25 , 58 , 58 , 58 , 26 , 58 , 27 , 28 , 29 , 30 , 58 , 31 , 58 , 58 , 58 , 32 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 33 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 34 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 35 , 36 , 37 , 58 , 38 , 58 , 58 , 58 , 39 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 40 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 41 , 42 , 43 , 58 , 44 , 58 , 58 , 58 , 45 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 46 , 47 , 48 , 58 , 49 , 58 , 58 , 58 , 50 , 51 , 52 , 58 , 53 , 54 , 55 , 56 , 57};
	unsigned int table_ri_8[256]     = {0 , 1 , 1 , 2 , 1 , 3 , 2 , 4 , 1 , 5 , 3 , 6 , 2 , 7 , 4 , 8 , 1 , 9 , 5 , 10 , 3 , 11 , 6 , 12 , 2 , 13 , 7 , 14 , 4 , 15,8,16,1,5,9,13,5,17,10,18,3,17,11,19,6,20,12,21,2,10,13,22,7,23,14,24,4,18,15,25,8,26,16,27,1,3,5,7,9,11,13,15,5,17,17,20,10,23,18,26,3,11,17,23,11,28,19,29,6,19,20,30,12,29,21,31,2,6,10,14,13,19,22,25,7,20,23,30,14,30,24,32,4,12,18,24,15,29,25,33,8,21,26,32,16,31,27,34,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,5,13,17,18,17,19,20,21,10,22,23,24,18,25,26,27,3,7,11,15,17,20,23,26,11,23,28,29,19,30,29,31,6,14,19,25,20,30,30,32,12,24,29,33,21,32,31,34,2,4,6,8,10,12,14,16,13,18,19,21,22,24,25,27,7,15,20,26,23,29,30,31,14,25,30,32,24,33,32,34,4,8,12,16,18,21,24,27,15,26,29,31,25,32,33,34,8,16,21,27,26,31,32,34,16,27,31,34,27,34,34,35};	
	unsigned int table_riu2_8[256]   = {0 , 1 , 1 , 2 , 1 , 9 , 2 , 3 , 1 , 9 , 9 , 9 , 2 , 9 , 3 , 4 , 1 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 2 , 9 , 9 , 9 , 3 , 9 , 4 , 5 , 1 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 2 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 3 , 9 , 9 , 9 , 4 , 9 , 5 , 6 , 1 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 2 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 3 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 4 , 9 , 9 , 9 , 5 , 9 , 6 , 7 , 1 , 2 , 9 , 3 , 9 , 9 , 9 , 4 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 5 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 6 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 7 , 2 , 3 , 9 , 4 , 9 , 9 , 9 , 5 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 6 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 7 , 3 , 4 , 9 , 5 , 9 , 9 , 9 , 6 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 7 , 4 , 5 , 9 , 6 , 9 , 9 , 9 , 7 , 5 , 6 , 9 , 7 , 6 , 7 , 7 , 8};

	unsigned int table_normal_4[16] = {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 };
	unsigned int table_u2_4[16]     = {0 , 1 , 2 , 3 , 4 , 14 , 5 , 6 , 7 , 8 , 14 , 9 , 10 , 11 , 12 , 13};
	unsigned int table_ri_4[16]     = {0 , 1 , 1 , 2 , 1 , 3 , 2 , 4 , 1 , 2 , 3 , 4 , 2 , 4 , 4 , 5};	
	unsigned int table_riu2_4[16]   = {0 , 1 , 1 , 2 , 1 , 5 , 2 , 3 , 1 , 2 , 5 , 3 , 2 , 3 , 3 , 4};
	unsigned int *table;

	Nbinsnscale                     = Nbins*nscale;
	NbinsnscalenH                   = Nbinsnscale*nH;

	if(cs_opt == 1)
	{
		powN                        = 16;
	}


	II                              = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	table                           = (unsigned int *) malloc((powN*(improvedLGP+1))*sizeof(unsigned int));

#ifdef OMP 

#else
	Itemp                           = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
#endif

#ifdef OMP 
    num_threads                     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif

	if(cs_opt == 1)
	{
		if(maptable == 0)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_normal_4[m]+v;
				}
			}
		}
		else if(maptable == 1)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				i = l*15;

				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_u2_4[m] + i;
				}
			}
		}
		else if(maptable == 2)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				i = l*6;
				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_ri_4[m] + i;
				}
			}
		}
		else if(maptable == 3)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				i = l*6;
				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_riu2_4[m] + i;
				}
			}
		}
	}
	else
	{
		if(maptable == 0)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_normal_8[m]+v;
				}
			}
		}
		else if(maptable == 1)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				i = l*59;
				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_u2_8[m] + i;
				}
			}
		}
		else if(maptable == 2)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				i = l*36;
				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_ri_8[m] + i;
				}
			}
		}
		else if(maptable == 3)
		{
			for (l = 0 ; l < improvedLGP+1 ; l++)
			{
				v = l*powN;
				i = l*10;

				for (m = 0 ; m < powN ; m++)
				{
					table[m + v]        = table_riu2_8[m] + i;
				}
			}
		}
	}


#ifdef OMP	
	Itemp                      = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
#endif	

	MakeIntegralImage(I , II , Nx , Ny , Itemp);

#ifdef OMP	
	free(Itemp);
#endif

	compute_mblgp(II , table , detector , Ny , Nx , Nbins , R);

#ifdef OMP 
#pragma omp parallel default(none) private(i,Itemp) shared(R,IIR,NyNx,Nx,Ny,Nbinsnscale)
#endif
	{
#ifdef OMP 
		Itemp                   = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
#else
#endif
#ifdef OMP 
#pragma omp for	nowait	
#endif
		for (i = 0 ; i < Nbinsnscale  ; i++)
		{		
			MakeIntegralImage(R + i*NyNx , IIR + i*NyNx , Nx , Ny , Itemp);
		}
#ifdef OMP
		free(Itemp);
#else
#endif
	}

	if(n > 0)
	{
		yest      = eval_hmblgp_spyr_subwindow_hom(IIR , H  , Ny , Nx  , Nbins , NbinsnscalenH  , maxfactor , detector , fx);
	}
	else
	{
		yest      = eval_hmblgp_spyr_subwindow(IIR , H  , Ny , Nx  , Nbins , NbinsnscalenH  , maxfactor , detector , fx);
	}


	yfx[0]    = (double) yest;

	free(II);
	free(table);

#ifdef OMP
#else
	free(Itemp);
#endif	

}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int eval_hmblgp_spyr_subwindow(unsigned int *IIR , double *H  , int Ny , int Nx , int Nbins , int  NbinsnscalenH  , double maxfactor , struct model detector  , double *fx)
{
	double *w = detector.w , *spyr = detector.spyr;
	double clamp = detector.clamp;
	double scaley, scalex,ratio , sum , temp;
	int nspyr = detector.nspyr , nscale = detector.nscale , rmextremebins = detector.rmextremebins , cs_opt = detector.cs_opt , improvedLGP = detector.improvedLGP;
	int p , l , m , s , i , j;
	int origy, origx, deltay, deltax, sy , sx , ly, lx , offsety = 0 , offsetx = 0 , coNbins = 0;
	int NyNx = Ny*Nx , NyNxNbins  , sNyNxNbins , NBINS;
	int co_p , co_totalp = 0 , Nbinsnscale = Nbins*nscale , offset , indj , indl;
	int norm_all = (int) detector.norm[0] , norm_p = (int) detector.norm[1] , norm_w = (int) detector.norm[2];

	if((improvedLGP == 1) && (cs_opt == 0))
	{
		NBINS       = Nbins/2;
	}

	NyNxNbins       = NyNx*Nbins;

	for (p = 0 ; p < nspyr ; p++)
	{
		scaley      = (spyr[p + nspyr*2]);
		ly          = (int) ( (1 - spyr[p + 0])/scaley + 1);
		deltay      = (int) (Ny*scaley);
		sy          = (int) (Ny*spyr[p + 0]);
		offsety     = max(0 , (int) ( floor(Ny - ( (ly-1)*deltay + sy + 1)) ));

		scalex      = (spyr[p + nspyr*3]);
		lx          = (int) ( (1 - spyr[p + nspyr*1])/scalex + 1);
		deltax      = (int) (Nx*scalex);
		sx          = (int) (Nx*spyr[p + nspyr*1]);
		offsetx     = max(0 , (int) ( floor(Nx - ( (lx-1)*deltax + sx + 1)) ));

		ratio       = 1.0/spyr[p + nspyr*4];
		co_p        = 0;
		offset      = co_totalp*Nbinsnscale;

		for(l = 0 ; l < lx ; l++) /* Loop shift on x-axis */
		{
			origx  = offsetx + l*deltax;
			for(m = 0 ; m < ly ; m++)   /* Loop shift on y-axis  */
			{
				origy     = offsety + m*deltay;
				for (s = 0 ; s < nscale ; s++)
				{
					sNyNxNbins   = s*NyNxNbins;

					for (i = 0 ; i < Nbins ; i++)
					{
						H[i + coNbins] = Area(IIR + i*NyNx + sNyNxNbins , origx   , origy   , sx , sy  , Ny);
					}
					for(i = coNbins ; i < coNbins+Nbins ; i++)
					{
						H[i]         *= ratio;
					}
					/* Normalization per subwindows */

					if(norm_w == 1)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							sum   += H[i];
						}
						sum = 1.0/(sum + tiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
						}
					}
					if(norm_w == 2)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							temp   = H[i];
							sum   += temp*temp;
						}
						sum = 1.0/sqrt(sum + verytiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
						}
					}
					if(norm_w == 3)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							sum   += H[i];
						}

						sum = 1.0/(sum + tiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   = sqrt(H[i]*sum);
						}
					}
					if(norm_w == 4)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							temp   = H[i];
							sum   += temp*temp;
						}
						sum = 1.0/sqrt(sum + verytiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
							if(H[i] > clamp)
							{
								H[i] = clamp;
							}
						}
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							temp   = H[i];
							sum   += temp*temp;
						}
						sum = 1.0/sqrt(sum + verytiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
						}
					}
					if(rmextremebins)
					{
						if(improvedLGP)
						{
							H[0 + coNbins] = H[NBINS-1 + coNbins] = H[NBINS + coNbins] = H[Nbins-1 + coNbins] = 0.0;
						}
						else
						{
							H[0 + coNbins] = H[Nbins-1 + coNbins] = 0.0;
						}
					}
					coNbins   += Nbins;
				}
				co_p++;
			}
		}
		/* Normalization per pyramid level */

		if(norm_p == 1)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						sum   += H[i];
					}
				}
				sum = 1.0/(sum + tiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i] *= sum;
					}
				}
			}
		}
		else if(norm_p == 2)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						temp   = H[i];
						sum   += temp*temp;
					}
				}
				sum = 1.0/sqrt(sum + verytiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i] *= sum;
					}
				}
			}
		}
		else if(norm_p == 3)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						sum   += H[i];
					}
				}
				sum = 1.0/(sum + tiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i]   = sqrt(H[i]*sum);
					}
				}
			}
		}
		else if(norm_p == 4)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						temp   = H[i];
						sum   += temp*temp;
					}
				}
				sum = 1.0/sqrt(sum + verytiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i]   *= sum;
						if(H[i] > clamp)
						{
							H[i] = clamp;
						}
					}
				}
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						temp   = H[i];
						sum   += temp*temp;
					}
				}
				sum = 1.0/sqrt(sum + verytiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i] *= sum;
					}
				}
			}
		}
		co_totalp       += co_p;
	}
	
	/* Normalization for full descriptor (NbinsnscalenH x 1) */
	if(norm_all > 0)
	{
		if(norm_all == 1)
		{
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				sum   += H[i];
			}
			sum = 1.0/(sum + tiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
			}
		}
		else if(norm_all == 2)
		{
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				temp   = H[i];
				sum   += temp*temp;
			}
			sum = 1.0/sqrt(sum + verytiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
			}
		}
		else if(norm_all == 3)
		{
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				sum   += H[i];
			}
			sum = 1.0/(sum + tiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   = sqrt(H[i]*sum);
			}
		}
		else if(norm_all == 4)
		{
			sum        = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				temp   = H[i];
				sum   += temp*temp;
			}
			sum = 1.0/sqrt(sum + verytiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
				if(H[i] > clamp)
				{
					H[i] = clamp;
				}
			}
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				temp   = H[i];
				sum   += temp*temp;
			}
			sum = 1.0/sqrt(sum + verytiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
			}
		}
	}

	sum   = 0.0;
	for(i = 0 ; i < NbinsnscalenH ; i++)
	{
		sum  += H[i]*w[i];
	}
	if(detector.addbias)
	{
		sum  += w[NbinsnscalenH];
	}

	fx[0] = sum;
	return (sign(sum));
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
int eval_hmblgp_spyr_subwindow_hom(unsigned int *IIR , double *H  , int Ny , int Nx , int Nbins , int  NbinsnscalenH  , double maxfactor , struct model detector  , double *fx)
{
	double *w = detector.w , *spyr = detector.spyr , *table = detector.homtable;
	double clamp = detector.clamp;
	double scaley, scalex,ratio , sum , temp;
	double mantissa  , f1 , f2;
	int nspyr = detector.nspyr , nscale = detector.nscale , rmextremebins = detector.rmextremebins , cs_opt = detector.cs_opt , improvedLGP = detector.improvedLGP;
	int maxexponent = detector.maxexponent , minexponent = detector.minexponent , numsubdiv = detector.numsubdiv;
	int n = detector.n , n1 = (2*n + 1) , numsubdivn1 = numsubdiv*n1;	
	int p , l , m , s , i , j;
	int origy, origx, deltay, deltax, sy , sx , ly, lx , offsety = 0 , offsetx = 0 , coNbins = 0;
	int NyNx = Ny*Nx , NyNxNbins  , sNyNxNbins , NBINS;
	int exponent;
	unsigned int v1 , v2 , co;
	double subdiv = 1.0 / numsubdiv;
	int norm_all = (int) detector.norm[0] , norm_p = (int) detector.norm[1] , norm_w = (int) detector.norm[2];
	int co_p , co_totalp = 0 , Nbinsnscale = Nbins*nscale , offset , indj , indl;


	if((improvedLGP == 1) && (cs_opt == 0))
	{
		NBINS       = Nbins/2;
	}

	NyNxNbins       = NyNx*Nbins;

	for (p = 0 ; p < nspyr ; p++)
	{
		scaley      = (spyr[p + nspyr*2]);
		ly          = (int) ( (1 - spyr[p + 0])/scaley + 1);
		deltay      = (int) (Ny*scaley);
		sy          = (int) (Ny*spyr[p + 0]);
		offsety     = max(0 , (int) ( floor(Ny - ( (ly-1)*deltay + sy + 1)) ));

		scalex      = (spyr[p + nspyr*3]);
		lx          = (int) ( (1 - spyr[p + nspyr*1])/scalex + 1);
		deltax      = (int) (Nx*scalex);
		sx          = (int) (Nx*spyr[p + nspyr*1]);
		offsetx     = max(0 , (int) ( floor(Nx - ( (lx-1)*deltax + sx + 1)) ));

		ratio       = 1.0/spyr[p + nspyr*4];
		co_p        = 0;
		offset      = co_totalp*Nbinsnscale;

		for(l = 0 ; l < lx ; l++) /* Loop shift on x-axis */
		{
			origx  = offsetx + l*deltax;

			for(m = 0 ; m < ly ; m++)   /* Loop shift on y-axis  */
			{
				origy     = offsety + m*deltay;

				for (s = 0 ; s < nscale ; s++)
				{
					sNyNxNbins   = s*NyNxNbins;

					for (i = 0 ; i < Nbins ; i++)
					{
						H[i + coNbins] = Area(IIR + i*NyNx + sNyNxNbins , origx   , origy   , sx , sy  , Ny);
					}

					for(i = coNbins ; i < coNbins+Nbins ; i++)
					{
						H[i]         *= ratio;
					}
					/* Normalization per subwindows */

					if(norm_w == 1)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							sum   += H[i];
						}
						sum = 1.0/(sum + tiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
						}
					}
					else if(norm_w == 2)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							temp   = H[i];
							sum   += temp*temp;
						}
						sum = 1.0/sqrt(sum + verytiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
						}
					}
					else if(norm_w == 3)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							sum   += H[i];
						}
						sum = 1.0/(sum + tiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   = sqrt(H[i]*sum);
						}
					}
					else if(norm_w == 4)
					{
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							temp   = H[i];
							sum   += temp*temp;
						}
						sum = 1.0/sqrt(sum + verytiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;

							if(H[i] > clamp)
							{
								H[i] = clamp;
							}
						}
						sum       = 0.0;
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							temp   = H[i];
							sum   += temp*temp;
						}
						sum = 1.0/sqrt(sum + verytiny);
						for(i = coNbins ; i < coNbins+Nbins ; i++)
						{
							H[i]   *= sum;
						}
					}

					if(rmextremebins)
					{
						if(improvedLGP)
						{
							H[0 + coNbins] = H[NBINS-1 + coNbins] = H[NBINS + coNbins] = H[Nbins-1 + coNbins] = 0.0;
						}
						else
						{
							H[0 + coNbins] = H[Nbins-1 + coNbins] = 0.0;
						}
					}
					coNbins   += Nbins;
				}
				co_p++;
			}
		}
		/* Normalization per pyramid level */
		if(norm_p == 1)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						sum   += H[i];
					}
				}
				sum = 1.0/(sum + tiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i] *= sum;
					}
				}
			}
		}
		else if(norm_p == 2)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						temp   = H[i];
						sum   += temp*temp;
					}
				}
				sum = 1.0/sqrt(sum + verytiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i] *= sum;
					}
				}
			}
		}
		else if(norm_p == 3)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						sum   += H[i];
					}
				}
				sum = 1.0/(sum + tiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i]   = sqrt(H[i]*sum);
					}
				}
			}
		}
		else if(norm_p == 4)
		{
			for (l = 0 ; l < nscale ; l++)
			{
				indl      = l*Nbins + offset;
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						temp   = H[i];
						sum   += temp*temp;
					}
				}
				sum = 1.0/sqrt(sum + verytiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i]   *= sum;
						if(H[i] > clamp)
						{
							H[i] = clamp;
						}
					}
				}
				sum       = 0.0;
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						temp   = H[i];
						sum   += temp*temp;
					}
				}
				sum = 1.0/sqrt(sum + verytiny);
				for(j = 0 ; j < co_p ; j++)
				{
					indj = j*Nbinsnscale + indl;
					for(i = indj ; i < (Nbins + indj) ; i++)
					{
						H[i] *= sum;
					}
				}
			}
		}

		co_totalp       += co_p;
	}
	/* Normalization for full descriptor (NbinsnscalenH x 1) */

	if(norm_all > 0)
	{
		if(norm_all == 1)
		{
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				sum   += H[i];
			}
			sum = 1.0/(sum + tiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
			}
		}
		else if(norm_all == 2)
		{
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				temp   = H[i];
				sum   += temp*temp;
			}
			sum = 1.0/sqrt(sum + verytiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
			}
		}
		else if(norm_all == 3)
		{
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				sum   += H[i];
			}
			sum = 1.0/(sum + tiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   = sqrt(H[i]*sum);
			}
		}
		else if(norm_all == 4)
		{
			sum        = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				temp   = H[i];
				sum   += temp*temp;
			}
			sum = 1.0/sqrt(sum + verytiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
				if(H[i] > clamp)
				{
					H[i] = clamp;
				}
			}
			sum       = 0.0;
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				temp   = H[i];
				sum   += temp*temp;
			}
			sum = 1.0/sqrt(sum + verytiny);
			for(i = 0 ; i < NbinsnscalenH ; i++)
			{
				H[i]   *= sum;
			}
		}
	}

	sum   = 0.0;
	co    = 0;
	for(i = 0 ; i < NbinsnscalenH ; i++)
	{
		mantissa  = frexp(H[i] , &exponent) ;
		mantissa *= 2 ;
		exponent -- ;

		if (mantissa == 0 || exponent <= minexponent || exponent >= maxexponent) 
		{
			co           += n1;
		}
		else
		{
			v1            = (exponent - minexponent) * numsubdivn1;
			mantissa     -= 1.0 ;
			while (mantissa >= subdiv) 
			{
				mantissa -= subdiv ;
				v1       += n1 ;
			}
			v2            = v1 + n1 ;
			for (l = 0 ; l < n1 ; ++l) 
			{
				f1           = table[l + v1];
				f2           = table[l + v2];
				sum         += ((f2 - f1) * (numsubdiv * mantissa) + f1)*w[co];
				co++;
			}
		}
	}
	if(detector.addbias)
	{
		sum  += w[NbinsnscalenH*n1];
	}

	fx[0] = sum;
	return (sign(sum));
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void compute_mblgp(unsigned int *II , unsigned int *table , struct model detector , int Ny , int Nx , int Nbins , unsigned char *R  )
{
	int s , xc , yc , xnw , ynw , xse , yse;
	int NyNx = Ny*Nx , NyNxNbins = NyNx*Nbins , sNyNxNbins , xcNy;
	double *scale = detector.scale;
	int nscale = detector.nscale , cs_opt = detector.cs_opt , improvedLGP = detector.improvedLGP;
	int currentscale ;
	double Ac , tmpA , sumA;
	double gn, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7;
	unsigned short int valF;

	if(cs_opt)
	{
		for (s = 0 ; s < nscale ; s++)
		{
			currentscale = (int) scale[s];
			sNyNxNbins   = s*NyNxNbins;

#ifdef OMP 
#pragma omp parallel for default(none) private(xc,yc,xnw,ynw,xse,yse,Ac,tmpA,sumA,valF,xcNy,gn,A0,A1,A2,A3) shared(II,R,table,Nx,Ny,NyNx,currentscale,sNyNxNbins,improvedLGP)
#endif
			for (xc = currentscale  ; xc <= Nx - 2*currentscale  ; xc++)
			{
				xcNy = xc*Ny + sNyNxNbins;

				for (yc = currentscale  ; yc <= Ny - 2*currentscale  ; yc++)
				{
					xnw   = xc - currentscale;
					ynw   = yc - currentscale;
					xse   = xc + currentscale;
					yse   = yc + currentscale;

					sumA  = 0.0;
					gn    = 0.0;

					Ac    = Area(II , xse , yse , currentscale , currentscale , Ny);
					tmpA  = Area(II , xnw , ynw , currentscale , currentscale , Ny);
					A0    = fabs(tmpA - Ac);
					gn   += A0;
					sumA += (Ac+tmpA);

					Ac    = Area(II , xc  , yse , currentscale , currentscale , Ny);
					tmpA  = Area(II , xc  , ynw , currentscale , currentscale , Ny);
					A1    = fabs(tmpA - Ac);
					gn   += A1;
					sumA += (Ac+tmpA);

					Ac    = Area(II , xnw , yse , currentscale , currentscale , Ny);
					tmpA  = Area(II , xse , ynw , currentscale , currentscale , Ny);
					A2    = fabs(tmpA - Ac);
					gn   += A2;
					sumA += (Ac+tmpA);

					Ac    = Area(II , xnw , yc , currentscale , currentscale , Ny);
					tmpA  = Area(II , xse , yc , currentscale , currentscale , Ny);
					A3    = fabs(tmpA - Ac);
					gn   += A3;
					sumA += (Ac+tmpA);

					gn   /= 4.0;

					valF  = 0;
					if(A0 > gn)
					{
						valF |= 0x01;
					}
					if(A1 > gn)
					{
						valF |= 0x02;
					}
					if(A2 > gn)
					{
						valF |= 0x04;
					}
					if(A3 > gn)
					{
						valF |= 0x08;
					}
					if(improvedLGP)
					{
						if(fabs(sumA -  (8.0*Area(II , xc  , yc  , currentscale , currentscale , Ny))) > 3.0*gn )
						{
							valF |= 0x10; 
						}
					}
					R[yc + xcNy + table[valF]*NyNx] = 1;
				}
			}
		}
	}
	else
	{
		for (s = 0 ; s < nscale ; s++)
		{
			currentscale = (int) scale[s];
			sNyNxNbins   = s*NyNxNbins;

#ifdef OMP 
#pragma omp parallel for default(none) private(xc,yc,xnw,ynw,xse,yse,Ac,tmpA,sumA,valF,xcNy,gn,A0,A1,A2,A3,A4,A5,A6,A7) shared(II,R,table,Nx,Ny,NyNx,currentscale,sNyNxNbins,improvedLGP)
#endif
			for (xc = currentscale  ; xc <= Nx - 2*currentscale  ; xc++)
			{
				xcNy = xc*Ny + sNyNxNbins;

				for (yc = currentscale  ; yc <= Ny - 2*currentscale  ; yc++)
				{
					xnw   = xc - currentscale;
					ynw   = yc - currentscale;
					xse   = xc + currentscale;
					yse   = yc + currentscale;

					sumA  = 0.0;
					gn    = 0.0;

					Ac    = Area(II , xc  , yc  , currentscale , currentscale , Ny);

					tmpA  = Area(II , xnw , ynw , currentscale , currentscale , Ny);
					A0    = fabs(tmpA - Ac);
					gn   += A0;
					sumA += tmpA;

					tmpA  = Area(II , xc  , ynw , currentscale , currentscale , Ny);
					A1    = fabs(tmpA - Ac);
					gn   += A1;
					sumA += tmpA;

					tmpA  = Area(II , xse , ynw , currentscale , currentscale , Ny);
					A2    = fabs(tmpA - Ac);
					gn   += A2;
					sumA += tmpA;

					tmpA  = Area(II , xse , yc  , currentscale , currentscale , Ny);
					A3    = fabs(tmpA - Ac);
					gn   += A3;
					sumA += tmpA;

					tmpA  = Area(II , xse , yse , currentscale , currentscale , Ny);
					A4    = fabs(tmpA - Ac);
					gn   += A4;
					sumA += tmpA;

					tmpA  = Area(II , xc  , yse , currentscale , currentscale , Ny);
					A5    = fabs(tmpA - Ac);
					gn   += A5;
					sumA += tmpA;

					tmpA  = Area(II , xnw , yse , currentscale , currentscale , Ny);
					A6    = fabs(tmpA - Ac);
					gn   += A6;
					sumA += tmpA;

					tmpA  = Area(II , xnw , yc , currentscale , currentscale , Ny);
					A7    = fabs(tmpA - Ac);						
					gn   += A7;
					sumA += tmpA;

					gn   /= 8.0;

					valF  = 0;
					if(A0 > gn)
					{
						valF |= 0x01;
					}
					if(A1 > gn)
					{
						valF |= 0x02;
					}
					if(A2 > gn)
					{
						valF |= 0x04;
					}
					if(A3 > gn)
					{
						valF |= 0x08;
					}
					if(A4 > gn)
					{
						valF |= 0x10;
					}
					if(A5 >= gn)
					{
						valF |= 0x20;
					}
					if(A6 > gn)
					{
						valF |= 0x40;
					}
					if(A7 > gn)
					{
						valF |= 0x80;
					}
					if(improvedLGP)
					{
						if(fabs(sumA  - 8.0*Ac) > 6.0*gn)
						{
							valF |= 0x100; 
						}
					}
					R[yc + xcNy + table[valF]*NyNx] = 1;
				}
			}
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------------*/
void MakeIntegralImage(unsigned char *pIn, unsigned int *pOut, int iXmax, int iYmax , unsigned int *pTemp)
{
	/* Variable declaration */
	int x , y , indx = 0;

	for(x=0 ; x< iXmax ; x++)
	{
		pTemp[indx]     = (unsigned int)pIn[indx];
		indx           += iYmax;
	}
	for(y = 1 ; y <iYmax ; y++)
	{
		pTemp[y]        = pTemp[y - 1] + (unsigned int)pIn[y];
	}
	pOut[0]             = (unsigned int)pIn[0];
	indx                = iYmax;
	for(x=1 ; x < iXmax ; x++)
	{
		pOut[indx]      = pOut[indx - iYmax] + pTemp[indx];
		indx           += iYmax;
	}
	for(y = 1 ; y < iYmax ; y++)
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
		indx           += iYmax;
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
int Round(double x)
{
	return ((int)(x + 0.5));
}
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
void qsindex (double  *a, int *index , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
   of the region of array a that is to be sorted 
*/

    int i=lo, j=hi , ind;
    double x=a[(lo+hi)/2] , h;
    /*  partition */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			ind      = index[i];
			index[i] = index[j];
			index[j] = ind;
            i++; 
			j--;
        }
    }
	while (i<=j);
    /*  recursion */
    if (lo<j) qsindex(a , index , lo , j);
    if (i<hi) qsindex(a , index , i , hi);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------- */
int	number_histo_lbp(double *spyr , int nspyr , int nscale)
{
	int l , nH = 0 , ly , lx ;

	for (l = 0 ; l < nspyr ; l++)
	{
		ly          = (int) ( (1 - spyr[l + 0]) /(spyr[l + nspyr*2]) + 1);  
		lx          = (int) ( (1 - spyr[l + nspyr*1])/(spyr[l + nspyr*3]) + 1); 
		nH         += (ly*lx);
	}
	return (nH*nscale);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void homkertable(struct model options , double *table )
{
	int n = options.n, kerneltype = options.kerneltype, numsubdiv = options.numsubdiv, minexponent = options.minexponent, maxexponent = options.maxexponent;
	double L = options.L , subdiv = 1.0 / numsubdiv;
	int exponent;
	unsigned int i,j,co=0;
	double x, logx, Lx, sqrtLx, Llogx, lambda ;
	double kappa, kappa0 , sqrtkappa0, sqrt2kappa ;
	double mantissa ;

	/* table initialization */

	if (kerneltype == 0)
	{
		kappa0          = 2.0/PI;
		sqrtkappa0      = sqrt(kappa0) ;
	}
	else if (kerneltype == 1)
	{
		kappa0          = 2.0/log(4.0);
		sqrtkappa0      = sqrt(kappa0) ;
	}
	else if (kerneltype == 2)
	{
		sqrtkappa0      = 1.0 ;
	}

	for (exponent  = minexponent ; exponent <= maxexponent ; ++exponent) 
	{
		mantissa        = 1.0;
		for (i = 0 ; i < numsubdiv ; ++i , mantissa += subdiv) 
		{
			x           = ldexp(mantissa, exponent);
			Lx          = L * x ;
			logx        = log(x);
			sqrtLx      = sqrt(Lx);
			Llogx       = L*logx;
			table[co++] = (sqrtkappa0 * sqrtLx);

			for (j = 1 ; j <= n ; ++j) 
			{
				lambda = j * L;
				if (kerneltype == 0)
				{
					kappa   = kappa0 / (1.0 + 4.0*lambda*lambda) ;
				}
				else if (kerneltype == 1)
				{
					kappa   = kappa0 * 2.0 / (exp(PI * lambda) + exp(-PI * lambda)) / (1.0 + 4.0*lambda*lambda) ;
				}
				else if (kerneltype == 2)
				{
					kappa   = 2.0 / (exp(PI * lambda) + exp(-PI * lambda)) ;
				}
				sqrt2kappa  = sqrt(2.0 * kappa)* sqrtLx ;
				table[co++] = (sqrt2kappa * cos(j * Llogx)) ;
				table[co++] = (sqrt2kappa * sin(j * Llogx)) ;
			}
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
