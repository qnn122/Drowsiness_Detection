
/*

  Compute HAAR features of image I

  Usage
  ------

  z                                     = haar(I , [options] );


  Inputs
  -------

  I                                     Images (Ny x Nx x N) in UINT8 format 

  options

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

         F                              Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current feature of the current pattern
										wf,hf  width and height of the current feature of the current pattern
										ir     Linear index of the FIRST rectangle of the current Haar feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle

         standardize                    Standardize Input Images 1 = yes, 0 = no (default = 1)

		 usesingle                      Output in single format if usesingle = 1 (default usesingle = 0)

		 transpose                      Transpose Output if tranpose = 1 (in order to speed up Boosting algorithm, default tranpose = 0)


If compiled with the "OMP" compilation flag

		 num_threads                    Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

   

  Outputs
  -------
  
  z                                     Haar features matrix (nF x N) or (N x nF) (acoording to transpose option) in single/double format (according to usesingle options)
                                        for each positions (y,x) in [1+h,...,ny-h]x[1+w,...,nx-w] and (w,h) integral block size.                              

  To compile
  ----------


  mex  -g -output haar.dll haar.c

  mex  -f mexopts_intel10.bat -output haar.dll haar.c

  If OMP directive is added, OpenMP support for multicore computation

  mex  -DOMP -f mexopts_intel10.bat -output haar.dll haar.c

  mex  -DOMP -f mexopts_intel10.bat -output haar.dll haar.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"



  Example 1
  ---------

  options = load('haar_dico_2.mat');

  Ny        = 24;
  Nx        = 24;
  P         = 10;
  N         = 8;
  I         = uint8(ceil(256*rand(Ny , Nx , P)));

  options.F = haar_featlist(Ny , Nx , options.rect_param);

  z         = haar(I , options);

  figure(1)
  imagesc(z)


  Example 2
  ---------

  clear, close all
  load viola_24x24
  options           = load('haar_dico_2.mat');
  options.transpose = 1;
  options.usesingle = 1;

  nF                = length(unique(options.rect_param(1 , :)));
  Ny                = 24;
  Nx                = 24;
  options.F         = haar_featlist(Ny , Nx , options.rect_param);
  pos               = find(y == 1);
  neg               = find(y == -1);

  I                 = (X(: , : , [pos(1:200) , neg(1:200)]));


  tic,z             = haar(I , options );,toc
  
  imagesc(z)
  title(sprintf('Haar feature''s with %d different patterns' , nF));
  colormap(gray)



  Example 3
  ---------


  clear, close all
  load viola_24x24
  options   = load('haar_dico_2.mat');

  nF        = length(unique(options.rect_param(1 , :)));
  [Ny,Nx,P] = size(X);
  Nimage    = 100;
  pattern   = [2 ; 1];


  FF        = haar_featlist(Ny , Nx , options.rect_param);

  ind       = find(sum(FF([4;5] , :) == repmat(pattern , 1 , size(FF , 2))) == 2);

  options.F = FF(: , ind);
  y         = options.F(3 , :);
  x         = options.F(2 , :);
  ind       = (y + 1 + (x)*Ny)';
 % index     = repmat(ind , 1 , P) + repmat((0:Ny*Nx:Ny*Nx*(P-1))' , 1 , P);
  
  z         = haar(X , options);

  Ihaar     = zeros(Ny , Nx , P , class(z));

  for i = 1:P
    
	  Ihaar(ind + (i-1)*Ny*Nx) = z(: , i); 
      Ihaar(: , : , i)         = Ihaar(: , : , i)';

  end

  figure
  display_database(X);
  title(sprintf('Original database'));

  figure
  display_database(Ihaar);
  title(sprintf('Haar feature''s with pattern [%d %d]' , pattern(1) , pattern(2)));



  Example 4
  ---------

  clear, close all
  load viola_24x24
  options   = load('haar_dico_2.mat');

  nP        = length(unique(options.rect_param(1 , :)));
  Ny        = 24;
  Nx        = 24;
  nF        = 1;
  FF        = haar_featlist(Ny , Nx , options.rect_param);
  indpos    = find(y == 1);
  indneg    = find(y == -1);

  options.F = FF(: , nF);


  tic,z     = haar(X , options);,toc
  
  plot(indpos , z(indpos) , indneg , z(indneg) , 'r')
  title(sprintf('Haar feature n=%d for %d different patterns' , nF , nP));
  legend('Faces' , 'Non-faces')



  Example 5
  ---------

  clear, close all
  load viola_24x24
  options   = load('haar_dico_2.mat');

  Ny        = 24;
  Nx        = 24;
  options.F = haar_featlist(Ny , Nx , options.rect_param);
  nF        = size(options.F , 2);

  indpos    = find(y==1);
  indneg    = find(y==-1);


  tic,zpos  = haar(X(: , : , indpos(1)) , options );,toc
  tic,zneg  = haar(X(: , : , indneg(1)) , options );,toc

  figure
  plot((1:nF)' , zpos , (1:nF)' , zneg , 'r')
  title(sprintf('Haar feature''s computed on Face/Non-Face image'));

  legend('Positive' , 'Negative')


  Example 6
  ---------


  clear, close all
  load viola_24x24
  options   = load('haar_dico_2.mat');

  nF        = length(unique(options.rect_param(1 , :)));
  [Ny,Nx,P] = size(X);
  pattern   = [6 ; 12];


  FF        = haar_featlist(Ny , Nx , options.rect_param);

  ind       = find(sum(FF([4;5] , :) == repmat(pattern , 1 , size(FF , 2))) == 2);

  indpos    = find(y==1);
  indneg    = find(y==-1);

  options.F = FF(: , ind(1));


  z         = haar(X , options);

  figure(1)
  plot(indpos , z(indpos) , indneg , z(indneg) , 'r')
  legend('Faces' , 'Non-faces')
  title(sprintf('Features = %d' , ind(1)))
  
  figure(2)
  hist(double(z(indneg)) , 100 , 'r' )
  set(get(gca , 'children') , 'facecolor' , [1 0 1])

  hold on
  hist(double(z(indpos)) , 100 )
  hold off
  legend(get(gca , 'children') , 'Faces' , 'Non-faces' )
  title(sprintf('Features = %d' , ind(1)))


  Example 7
  ---------


  clear, close all
  load viola_24x24
  options   = load('haar_dico_2.mat');

  nF        = length(unique(options.rect_param(1 , :)));
  [Ny,Nx,P] = size(X);


  FF        = haar_featlist(Ny , Nx , options.rect_param);

  indpos    = find(y==1);
  indneg    = find(y==-1);
  bestFeat  = 10992; %40478+1;
  

  options.F = FF(: , bestFeat);

  z         = haar(X , options);

  figure(1)
  plot(indpos , z(indpos) , indneg , z(indneg) , 'r')
  legend('Faces' , 'Non-faces')
  title(sprintf('Features = %d' , bestFeat))
  
  figure(2)
  hist(double(z(indneg)) , 100 , 'r' )
  set(get(gca , 'children') , 'facecolor' , [1 0 1])

  hold on
  hist(double(z(indpos)) , 100 )
  hold off
  legend(get(gca , 'children') , 'Faces' , 'Non-faces' )
  title(sprintf('Feature = %d' , bestFeat))



  Example 8
  ---------


  clear, close all
  load viola_24x24
  options   = load('haar_dico_2.mat');

  nP        = length(unique(options.rect_param(1 , :)));
  [Ny,Nx,P] = size(X);
  Nimage    = 100;

  I         = (X(: , : , Nimage));


  options.F = haar_featlist(Ny , Nx , options.rect_param);
  z         = haar(I , options);
  plot(z)



 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/20/2009


 Changelog :  - Add OpenMP support
 ----------   - Add double/single output format
              - Add transpose output

 References : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ----------       The annals of statistics, 1999

              [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
			      ICB07

			  [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
			  [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004
*/

#include <math.h>
#include <mex.h>

#ifdef OMP 
 #include <omp.h>
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

struct opts
{
	double        *rect_param;
	int            nR;

	unsigned int  *F;
	int            nF;

	int            standardize;
	int            usesingle;
	int            transpose;

#ifdef OMP 
    int            num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
void MakeIntegralImage(unsigned char *, unsigned int  *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
void shaar(unsigned char * , int , int , int , struct opts , float *);
void dhaar(unsigned char * , int , int , int , struct opts , double *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{    
	unsigned char *I;
	const int *dimsI;
	int numdimsI;
	struct opts options;   
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
    int i , Ny , Nx , N = 1 , tempint;
	double *zd , *tmp;
	float *zs;
	int numdimsz = 2;
	int *dimsz;
	
	mxArray *mxtemp;
	
    options.nR           = 4;
    options.nF           = 0;
	options.standardize  = 1;
	options.usesingle    = 0;
	options.transpose    = 0;
#ifdef OMP 
    options.num_threads  = -1;
#endif
 
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
		mexErrMsgTxt("I must be (Ny x Nx x N) in UINT8 format");
	}
    
    /* Input 2  */

	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
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

		mxtemp                            = mxGetField( prhs[1] , 0, "standardize" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("standardize = {0,1}, force to 1");		
				options.standardize      = 1;
			}
			else
			{
				options.standardize      = tempint;	
			}		
		}
		
		mxtemp                            = mxGetField( prhs[1] , 0, "usesingle" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);			
			tempint                       = (int) tmp[0];

			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("usesingle = {0,1}, force to 0");		
				options.usesingle      = 0;
			}
			else
			{
				options.usesingle      = tempint;	
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
				options.transpose      = 0;
			}
			else
			{
				options.transpose      = tempint;	
			}			
		}
#ifdef OMP 
		
		mxtemp                            = mxGetField( prhs[1] , 0, "num_threads" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];	
			if((tempint < -2))
			{								
				options.num_threads       = -1;
			}
			else
			{
				options.num_threads       = tempint;	
			}			
		}
#endif
	}
    else
    {
		options.rect_param            = (double *)mxMalloc(40*sizeof(double));	
		for (i = 0 ; i < 40 ; i++)
		{		
			options.rect_param[i]     = rect_param_default[i];
		}		

		options.nF                    = number_haar_features(Ny , Nx , options.rect_param , options.nR);
		options.F                     = (unsigned int *)mxMalloc(5*options.nF*sizeof(unsigned int));
		haar_featlist(Ny , Nx , options.rect_param , options.nR , options.F);	        
    }   
	
    /*------------------------ Output ----------------------------*/
	if(options.transpose)
	{
		dimsz         = (int *)mxMalloc(2*sizeof(int));
		dimsz[0]      = N;
		dimsz[1]      = options.nF;
	}
	else
	{
		dimsz         = (int *)mxMalloc(2*sizeof(int));
		dimsz[0]      = options.nF;
		dimsz[1]      = N;
	}
	if(options.usesingle == 1)
	{
		plhs[0]       = mxCreateNumericArray(numdimsz , dimsz , mxSINGLE_CLASS , mxREAL);
		zs            = (float *)mxGetPr(plhs[0]);

		/*------------------------ Main Call ----------------------------*/
		shaar(I , Ny , Nx , N , options ,  zs);
	}
	else
	{
		plhs[0]       = mxCreateNumericArray(numdimsz , dimsz , mxDOUBLE_CLASS , mxREAL);
		zd            = mxGetPr(plhs[0]);

		/*------------------------ Main Call ----------------------------*/

		dhaar(I , Ny , Nx , N , options ,  zd);
	}
	/*----------------- Free Memory --------------------------------*/		
	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "rect_param" )) == NULL )
		{
			mxFree(options.rect_param);
		}
		if ( (mxGetField( prhs[1] , 0 , "F" )) == NULL )
		{
			mxFree(options.F);
		}
	}
	else
	{
		mxFree(options.rect_param);
		mxFree(options.F);
	}
	mxFree(dimsz);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dhaar(unsigned char *I , int Ny , int Nx , int P , struct opts options , double *z)
{
	double *rect_param = options.rect_param;
	unsigned int *F = options.F;
	int nF = options.nF , standardize = options.standardize , transpose = options.transpose;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	int  p , indF , NxNy = Nx*Ny , indNxNy = 0 ;
	int f , indnF = 0;
	int x , xr , y , yr  , w , wr  , h , hr , i , r , R , indR ;
	int coeffw , coeffh;
	int last;
	double val , s  , var , mean , std , cteNxNy;
	unsigned int *II , *Itemp , tempI;

	II              = (unsigned int *)malloc(NxNy*sizeof(unsigned int));
	Itemp           = (unsigned int *)malloc(NxNy*sizeof(unsigned int));

#ifdef OMP 
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif

	if(standardize)
	{
		cteNxNy     = 1.0/NxNy;
		last        = NxNy - 1;
/*
#ifdef OMP 
#pragma omp parallel for default(none) private(p,i,f,r,tempI,mean,std,x,y,w,h,R,coeffw,coeffh,xr,yr,wr,hr,s,var) shared(z,I,II,Itemp,F,rect_param,nF,P,Nx,Ny,NxNy,last,cteNxNy) reduction (*:indF,indnF,indNxNy)  reduction (+:val,indR) 
#endif
*/
		for(p = 0 ; p < P ; p++)
		{	
			indNxNy    = p*NxNy;
			indnF      = p*nF;	

			MakeIntegralImage(I + indNxNy , II , Nx , Ny , Itemp);

			var        = 0.0;
			for(i = 0 ; i < NxNy ; i++)
			{
				tempI      = I[i + indNxNy];
				var       += (tempI*tempI);
			}

			var       *= cteNxNy;
			mean      = II[last]*cteNxNy;
			std       = 1.0/sqrt(var - mean*mean);

#ifdef OMP 
#pragma omp parallel for default(none) private(f,r,x,y,w,h,R,coeffw,coeffh,xr,yr,wr,hr,s) shared(p,P,transpose,z,II,F,rect_param,nF,Nx,Ny,std,indnF) reduction (*:indF)  reduction (+:val,indR) 
#endif
			for (f = 0 ; f < nF ; f++)
			{	
				indF  = f*6;
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
					xr      = x + coeffw*(int)rect_param[5 + indR];
					yr      = y + coeffh*(int)rect_param[6 + indR];
					wr      = coeffw*(int)(rect_param[7 + indR]);
					hr      = coeffh*(int)(rect_param[8 + indR]);
					s       = rect_param[9 + indR];
					val    += s*Area(II,xr,yr,wr,hr,Ny);
					indR   += 10;
				}
				if(transpose)
				{
					z[p + f*P]    = val*std;	
				}
				else
				{
					z[f + indnF]  = val*std;	
				}
			}
		}
	}
	else
	{
		for(p = 0 ; p < P ; p++)
		{
			indNxNy    = p*NxNy;
			indnF      = p*nF;	
			
			MakeIntegralImage(I + indNxNy , II , Nx , Ny , Itemp);	

#ifdef OMP 
#pragma omp parallel for default(none) private(f,r,x,y,w,h,R,coeffw,coeffh,xr,yr,wr,hr,s) shared(p,P,transpose,z,II,F,rect_param,nF,Nx,Ny,std,indnF) reduction (*:indF)  reduction (+:val,indR) 
#endif
			for (f = 0 ; f < nF ; f++)
			{
				indF  = f*6;
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
					xr      = x + coeffw*(int)rect_param[5 + indR];
					yr      = y + coeffh*(int)rect_param[6 + indR];		
					wr      = coeffw*(int)(rect_param[7 + indR]);
					hr      = coeffh*(int)(rect_param[8 + indR]);
					s       = rect_param[9 + indR];
					val    += s*Area(II , xr  , yr  , wr , hr , Ny);
					indR   += 10;
				}
				if(transpose)
				{
					z[p + f*P]    = val;	
				}
				else
				{
					z[f + indnF]    = val;	
				}
			}
		}
	}
	free(II);
	free(Itemp);	
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void shaar(unsigned char *I , int Ny , int Nx , int P , struct opts options , float *z)
{
	double *rect_param = options.rect_param;
	unsigned int *F = options.F;
	int nF = options.nF , standardize = options.standardize , transpose = options.transpose;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	int  p , indF , NxNy = Nx*Ny , indNxNy = 0 ;
	int f , indnF = 0;
	int x , xr , y , yr  , w , wr  , h , hr , i , r , R , indR ;
	int coeffw , coeffh;
	int last;
	double val , s  , var , mean , std , cteNxNy;
	unsigned int *II , *Itemp , tempI;

	II              = (unsigned int *)malloc(NxNy*sizeof(unsigned int));
	Itemp           = (unsigned int *)malloc(NxNy*sizeof(unsigned int));

#ifdef OMP 
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif

	if(standardize)
	{
		cteNxNy     = 1.0/NxNy;
		last        = NxNy - 1;
/*
#ifdef OMP 
#pragma omp parallel for default(none) private(p,i,f,r,tempI,mean,std,x,y,w,h,R,coeffw,coeffh,xr,yr,wr,hr,s,var) shared(z,I,II,Itemp,F,rect_param,nF,P,Nx,Ny,NxNy,last,cteNxNy) reduction (*:indF,indnF,indNxNy)  reduction (+:val,indR) 
#endif
*/
		for(p = 0 ; p < P ; p++)
		{	
			indNxNy    = p*NxNy;
			indnF      = p*nF;	

			MakeIntegralImage(I + indNxNy , II , Nx , Ny , Itemp);

			var        = 0.0;
			for(i = 0 ; i < NxNy ; i++)
			{
				tempI      = I[i + indNxNy];
				var       += (tempI*tempI);
			}

			var       *= cteNxNy;
			mean      = II[last]*cteNxNy;
			std       = 1.0/sqrt(var - mean*mean);

#ifdef OMP 
#pragma omp parallel for default(none) private(f,r,x,y,w,h,R,coeffw,coeffh,xr,yr,wr,hr,s) shared(transpose,p,P,z,II,F,rect_param,nF,Nx,Ny,std,indnF) reduction (*:indF)  reduction (+:val,indR) 
#endif

			for (f = 0 ; f < nF ; f++)
			{	
				indF  = f*6;
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
					xr      = x + coeffw*(int)rect_param[5 + indR];
					yr      = y + coeffh*(int)rect_param[6 + indR];
					wr      = coeffw*(int)(rect_param[7 + indR]);
					hr      = coeffh*(int)(rect_param[8 + indR]);
					s       = rect_param[9 + indR];
					val    += s*Area(II,xr,yr,wr,hr,Ny);
					indR   += 10;
				}
				if(transpose)
				{
					z[p + f*P]    = (float)(val*std);	
				}
				else
				{
					z[f + indnF]  = (float)(val*std);	
				}
			}
		}
	}
	else
	{
		for(p = 0 ; p < P ; p++)
		{
			indNxNy    = p*NxNy;
			indnF      = p*nF;	
			
			MakeIntegralImage(I + indNxNy , II , Nx , Ny , Itemp);	

#ifdef OMP 
#pragma omp parallel for default(none) private(f,r,x,y,w,h,R,coeffw,coeffh,xr,yr,wr,hr,s) shared(transpose,p,P,z,II,F,rect_param,nF,Nx,Ny,std,indnF) reduction (*:indF)  reduction (+:val,indR) 
#endif
			for (f = 0 ; f < nF ; f++)
			{
				indF  = f*6;
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
					xr      = x + coeffw*(int)rect_param[5 + indR];
					yr      = y + coeffh*(int)rect_param[6 + indR];		
					wr      = coeffw*(int)(rect_param[7 + indR]);
					hr      = coeffh*(int)(rect_param[8 + indR]);
					s       = rect_param[9 + indR];
					val    += s*Area(II , xr  , yr  , wr , hr , Ny);
					indR   += 10;
				}
				if(transpose)
				{
					z[p + f*P]    = (float)val;	
				}
				else
				{
					z[f + indnF]  = (float)val;	
				}
			}
		}
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
/*----------------------------------------------------------------------------------------------------------------------------------------------*/
