
/*

  (Center-Symetric) MultiBlock Local Binary Pattern

  Usage
  ------

  z         =     mblbp(I , [options]);

  
  Inputs
  -------

  I                                    Images (Ny x Nx x N) in UINT8 format (unsigned char)

  options

          F                            Features lists (5 x nF) int UINT32 where nF design the total number of mblbp features (see mblbp_featlist function)
                                       F(: , i) = [if ; xf ; yf ; wf ; hf] where
									   if       index of the current feature, if = [1,...,nF]
									   xf,yf    coordinates of the current feature (top-left rectangle)
                                       wf,hf    width and height of each of the 9 rectangles
          map                          Feature's mapping vector in UINT8 format (unsigned char) (default map = 0:255)
          cs_opt                       Center-Symetric LBP : 1 for computing CS-MBLBP features, 0 : for MBLBP (default cs_opt = 0)
          a                            Tolerance (default a = 0)
		  transpose                    Transpose Output if tranpose = 1 (in order to speed up Boosting algorithm, default tranpose = 0)

If compiled with the "OMP" compilation flag
	     num_threads                   Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  z                                    MultiBlock LPB vector (nF x N) in UINT8 format for each positions (y,x) in [1+h,...,ny-h]x[1+w,...,nx-w] and (w,h) integral block size.                              

  To compile
  ----------

  mex  -g -output mblbp.dll mblbp.c

  mex  -f mexopts_intel10.bat -output mblbp.dll mblbp.c

  If OMP directive is added, OpenMP support for multicore computation

  mex  -v -DOMP -f mexopts_intel10.bat -output mblbp.dll mblbp.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


  Example 1  LBP, LBP_{u2} and LBP_{riu2}
  ---------


  Ny          = 24;
  Nx          = 24;
  P           = 10;
  N           = 8;
  scale       = [1 ; 1];
  I           = uint8(floor(256*rand(Ny , Nx , P)));

  options.F   = mblbp_featlist(Ny , Nx , scale);

  lbp         = mblbp(I , options);

  mapping     = getmapping(N,'u2');
  options.map = uint8(mapping.table);

  lbpu2       = mblbp(I , options);


  mapping     = getmapping(N,'riu2');
  options.map = uint8(mapping.table);

  lbpriu2     = mblbp(I , options);



  Example 2
  ---------

  clear, close all
  Ny               = 24;
  Nx               = 24;
  N                = 8;
  Nimage           = 200;

  scale            = [1 , 2 ; 1 , 2];
  options.cs_opt   = 1;

  load viola_24x24

  options.F        = mblbp_featlist(Ny , Nx , scale);
  mapping1         = getmapping(N,'u2');
  options.map      = uint8(mapping1.table);
  %options.map      = uint8(0:2^N-1);


  I                = X(: , : , Nimage);
  z1               = mblbp(I , options);

  mapping2         = getmapping(N/2,'u2');
  options.map      = uint8(mapping2.table);
  %options.map      = uint8(0:2^N/2-1);


  z2               = mblbp(I , options );


  ind1             = find(options.F(1 , :) == 1);
  template1        = options.F(: , ind1(1));
  Nxx1             = (Nx-3*template1(4) + 1);
  Nyy1             = (Ny-3*template1(5) + 1);
  Imblbp10         = reshape(z1(ind1) , Nyy1 , Nxx1);
  Imblbp11         = reshape(z2(ind1) , Nyy1 , Nxx1);


  ind2             = find(options.F(1 , :) == 2);
  template2        = options.F(: , ind2(1));
  Nxx2             = (Nx-3*template2(4) + 1);
  Nyy2             = (Ny-3*template2(5) + 1);
  Imblbp20         = reshape(z1(ind2) , Nyy2 , Nxx2);
  Imblbp21         = reshape(z2(ind2) , Nyy2 , Nxx2);


  figure(1)
  subplot(231)
  imagesc(I);
  title('Original Image');
  colorbar
  axis square

  subplot(232)
  imagesc(imresize(Imblbp10 , [Ny , Nx]))
  title(sprintf('MBLBP with s = %d' , scale(1,1)));
  colorbar
  axis square

  subplot(233)
  imagesc(imresize(Imblbp20 , [Ny , Nx]))
  title(sprintf('MBLBP with s = %d' , scale(1,2)));
  colorbar
  axis square


  subplot(234)
  imagesc(I);
  title('Original Image');
  colorbar
  axis square

  subplot(235)
  imagesc(imresize(Imblbp11 , [Ny , Nx]))
  title(sprintf('CSMBLBP with s = %d' , scale(1,1)));
  colorbar
  axis square

  subplot(236)
  imagesc(imresize(Imblbp21 , [Ny , Nx]))
  title(sprintf('CSMBLBP with s = %d' , scale(1,2)));
  colorbar
  axis square
  colormap(gray)


  Example 3
  ---------

  clear, close all
  I                   = imread('0000_-12_0_0_15_0_1.pgm');
  [Ny , Nx]           = size(I);
  N                   = 8;
  scale               = 5*[1  ; 1 ];
  options.cs_opt      = 1;

  options.F           = mblbp_featlist(Ny , Nx , scale);


  mapping1            = getmapping(N,'u2');
  %options.map         = uint8(mapping1.table);
  options.map         = uint8(0:2^N-1);

  z1                  = mblbp(I , options);


  mapping2            = getmapping(N/2,'u2');
  %options.map         = uint8(mapping2.table);
  options.map2        = uint8(0:2^N/2-1);


  z2                  = mblbp(I , options);

  template           = options.F(: , 1);

  Nxx                = (Nx-3*template(4) + 1);
  Nyy                = (Ny-3*template(5) + 1);

  Imblbp1            = reshape(z1 , Nyy , Nxx);
  Imblbp2            = reshape(z2 , Nyy , Nxx);

  figure(1)
  
  subplot(131)
  imagesc(I);
  title('Original Image');
  axis square


  subplot(132)
  imagesc(imresize(Imblbp1 , [Ny , Nx]))
  title(sprintf('MBLBP with s = %d' , scale(1)));
  axis square

  subplot(133)
  imagesc(imresize(Imblbp2 , [Ny , Nx]))
  title(sprintf('CSMBLBP with s = %d' , scale(1)));
  axis square
  colormap(gray)



  Example 4
  ---------

  clear, close all
  load viola_24x24

  [Ny,Nx,P]         = size(X);
  N                 = 8;
  scale             = 1*[1  ; 1 ];

 % mapping          = getmapping(N,'u2');
 % options.map      = uint8(mapping.table);
 options.map        = uint8(0:2^N-1);


  options.F         = mblbp_featlist(Ny , Nx , scale);
  z                 = mblbp(X , options);

  template          = options.F(: , 1);

  Nxx               = (Nx-3*template(4) + 1);
  Nyy               = (Ny-3*template(5) + 1);

  Xmlbp     = zeros(Ny , Nx , P , class(z));
  for i = 1:P
	  I                        = reshape(z(: , i) , [Nyy , Nxx]); 
      Xmlbp(: , : , i)         = imresize(I , [Ny , Nx]);
  end

  figure
  display_database(X);
  
  figure
  display_database(Xmlbp);
  title(sprintf('MLBP feature''s with scale %d' , scale));


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 05/02/2009


 Changelog : - Add Center-Symetric option
 ----------  - Add openMP support
             - Add transpose option

 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999

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
	int           cs_opt ;
	unsigned int  a;
	unsigned int  *F;
	int           nF;
	unsigned char *map;
	int            transpose;
#ifdef OMP 
    int            num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

int number_mblbp_features(int , int );
void mblbp_featlist(int  , int , unsigned int *);
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
void mblbp(unsigned char * , int , int , int , struct opts , unsigned char *);

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *I;
	const int *dimsI;
	int numdimsI;
	struct opts options;
    int Ny , Nx;
	int P = 1 , powN  = 256 , i;
	unsigned char *z;
	int numdimsz = 2;
	int *dimsz;
	mxArray *mxtemp;
	double *tmp;
	int tempint;

	options.cs_opt       = 0;
	options.a            = 0;
	options.transpose    = 0;

#ifdef OMP 
    options.num_threads  = -1;
#endif
   
    /* Input 1  */   
    if ((nrhs > 0) && !mxIsEmpty(prhs[0]) && mxIsUint8(prhs[0]))   
    {        
		dimsI    = mxGetDimensions(prhs[0]);
        numdimsI = mxGetNumberOfDimensions(prhs[0]);
		I        = (unsigned char *)mxGetData(prhs[0]);
		Ny       = dimsI[0];
		Nx       = dimsI[1];
		if(numdimsI > 2)
		{
			P    = dimsI[2];
		}    
    }

    /* Input 2  */

    if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
    {
		mxtemp     = mxGetField(prhs[1] , 0 , "cs_opt");
		if(mxtemp != NULL)
		{	
			tmp                           = mxGetPr(mxtemp);		
			tempint                       = (int) tmp[0];	
			if((tempint < 0) || (tempint > 1))
			{
				mexPrintf("cs_opt = {0,1}, force to 0");	
				options.cs_opt            = 0;			
			}
			else
			{
				options.cs_opt            = tempint;	
			}
			if(options.cs_opt == 1)
			{
				powN                      = 16;
			}
		}
			
		mxtemp     = mxGetField(prhs[1] , 0 , "a");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (unsigned int) tmp[0];
			if((tempint < 0) )
			{
				mexPrintf("a >= 0");	
				options.a                 = 0;			
			}
			else
			{	
				options.a                 = tempint;			
			}	
		}

		mxtemp     = mxGetField(prhs[1] , 0 , "F");
		if( (mxtemp != NULL) && (mxIsUint32(mxtemp)) )
		{         
			options.F                     = (unsigned int *) mxGetData(mxtemp);
			options.nF                    = mxGetN(mxtemp);
		}
		else
		{
			options.nF                    = number_mblbp_features(Ny , Nx);
			options.F                     = (unsigned int *)mxMalloc(5*options.nF*sizeof(int));
			mblbp_featlist(Ny , Nx , options.F);
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "map");
		if( (mxtemp != NULL) && (mxIsUint8(mxtemp)) )
		{        
			options.map                   = (unsigned char *) mxGetData(mxtemp);
		}
		else
		{
			options.map                   = (unsigned char *)mxMalloc(powN*sizeof(unsigned char));
			for(i = 0 ; i < powN ; i++)
			{
				options.map[i]  = i;
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
				options.transpose         = 0;
			}
			else
			{
				options.transpose         = tempint;	
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
    
    /* Output 1  */

	dimsz         = (int *)mxMalloc(2*sizeof(int));
	if(options.transpose)
	{
		dimsz[0]      = P;
		dimsz[1]      = options.nF;
	}
	else
	{
		dimsz[0]      = options.nF;
		dimsz[1]      = P;
	}
	
	plhs[0]       = mxCreateNumericArray(numdimsz , dimsz , mxUINT8_CLASS , mxREAL);
	z             = (unsigned char *)mxGetPr(plhs[0]);
	    
    /*------------------------ Main Call ----------------------------*/
	
	mblbp(I , Ny , Nx , P , options , z);

	/*----------------- Free Memory --------------------------------*/
	
	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "F" )) == NULL )
		{
			mxFree(options.F);
		}
		if ( (mxGetField( prhs[1] , 0 , "map" )) == NULL )
		{
			mxFree(options.map);
		}
	}
	else
	{
		mxFree(options.F);
		mxFree(options.map);
	}

	mxFree(dimsz);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void mblbp(unsigned char *I , int Ny , int Nx , int P , struct opts options, unsigned char *z)
{	
	unsigned char *map = options.map;
	unsigned int *F = options.F;
	unsigned int a = options.a , nF = options.nF , cs_opt = options.cs_opt , transpose = options.transpose;
#ifdef OMP 
    int num_threads = options.num_threads;
#endif
	int i , p , indF , NyNx = Ny*Nx , indnF = 0;
	int xc , yc , xnw , ynw , xse , yse   , w , h;
	unsigned int Ac ;
	unsigned int *II , *Itemp;
	unsigned char valF;
	
	II          = (unsigned int *)malloc(NyNx*sizeof(unsigned int));
	Itemp       = (unsigned int *)malloc(NyNx*sizeof(unsigned int));

#ifdef OMP 
    num_threads          = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif
	
	if(cs_opt == 0)
	{
		for(p = 0 ; p < P ; p++)
		{
			indnF       = p*nF;
			MakeIntegralImage(I + p*NyNx , II , Nx , Ny , Itemp);

#ifdef OMP 
#pragma omp parallel for default(none) private(i,indF,xc,yc,w,h,xnw,ynw,xse,yse,Ac,valF) shared(transpose,p,P,nF,F,II,map,z,Ny,a,indnF)
#endif
			for (i = 0 ; i < nF ; i++)
			{
				indF  = i*5;
				xc    = F[1 + indF];
				yc    = F[2 + indF];				
				w     = F[3 + indF];
				h     = F[4 + indF];

				xnw   = xc - w;
				ynw   = yc - h;
				xse   = xc + w;
				yse   = yc + h;

				Ac    = Area(II , xc  , yc  , w , h , Ny);

				valF  = 0;
				if(Area(II , xnw , ynw , w , h , Ny) + a > Ac)
				{
					valF |= 0x01;		
				}
				if(Area(II , xc  , ynw , w , h , Ny) + a > Ac)
				{
					valF |= 0x02;	
				}
				if(Area(II , xse , ynw , w , h , Ny) + a > Ac)
				{
					valF |= 0x04;		
				}
				if(Area(II , xse , yc  , w , h , Ny) + a > Ac)
				{
					valF |= 0x08;
				}
				if(Area(II , xse , yse , w , h , Ny) + a > Ac)
				{
					valF |= 0x10;		
				}
				if(Area(II , xc  , yse , w , h , Ny) + a > Ac)
				{
					valF |= 0x20;	
				}
				if(Area(II , xnw , yse , w , h , Ny) + a > Ac)
				{
					valF |= 0x40;				
				}
				if(Area(II , xnw , yc  , w , h , Ny) + a > Ac)
				{
					valF |= 0x80;
				}
				if(transpose)
				{
					z[p + i*P]     = map[valF];
				}
				else
				{
					z[i + indnF]   = map[valF];
				}
			}
		}	
	}
	if(cs_opt == 1)	
	{
		for(p = 0 ; p < P ; p++)
		{
			indnF       = p*nF;
			MakeIntegralImage(I + p*NyNx , II , Nx , Ny , Itemp);

#ifdef OMP 
#pragma omp parallel for default(none) private(i,indF,xc,yc,w,h,xnw,ynw,xse,yse,Ac,valF) shared(transpose,p,P,nF,F,II,map,z,Ny,a,indnF)
#endif
			for (i = 0 ; i < nF ; i++)
			{				
				indF  = i*5;

				xc    = F[1 + indF];
				yc    = F[2 + indF];

				w     = F[3 + indF];
				h     = F[4 + indF];

				xnw   = xc - w;
				ynw   = yc - h;	
				xse   = xc + w;
				yse   = yc + h;

				valF  = 0;
				if(Area(II , xnw , ynw , w , h , Ny) + a > Area(II , xse , ynw , w , h , Ny))
				{
					valF |= 0x01;					
				}
				if(Area(II , xc  , ynw , w , h , Ny) + a > Area(II , xc  , yse , w , h , Ny))
				{
					valF |= 0x02;
				}
				if(Area(II , xse , ynw , w , h , Ny) + a > Area(II , xnw , yse , w , h , Ny))
				{
					valF |= 0x04;
				}
				if(Area(II , xse , yc  , w , h , Ny) + a > Area(II , xnw , yc  , w , h , Ny))
				{
					valF |= 0x08;
				}	
				if(transpose)
				{
					z[p + i*P]    = map[valF];
				}
				else
				{
					z[i + indnF]  = map[valF];
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
void mblbp_featlist(int ny , int nx , unsigned int *F)
{
	int i , j , w = 1 , h , nofeat = 1 , co = 0; 
		
	while(nx >= 3*w)
	{		
		h    = 1;
		
		while(ny >= 3*h)
		{
			for (j = w ; j <= nx-2*w ; j++)
			{
				for (i = h ; i <= ny-2*h ; i++)
				{		
					F[0 + co] = nofeat;	
					F[1 + co] = j;
					F[2 + co] = i;
					F[3 + co] = w;
					F[4 + co] = h;
					co       += 5;
				}		
			}
			h++;
			nofeat++;		
		}
		w++;	
	}	
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int number_mblbp_features(int ny , int nx)
{
	int nF = 0 , X , Y  , nx1 = nx + 1 , ny1 = ny + 1 ;

	X           = (int) floor(nx/3);
	Y           = (int) floor(ny/3);
	nF          = (int) (X*Y*(nx1 - (X+1)*1.5)*(ny1 - (Y+1)*1.5));
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
