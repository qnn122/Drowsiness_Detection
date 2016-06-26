
/*

Return the Circular Histogram of Local Binary Patterns for image I.

Usage
------

H = chlbp(I , [options] );

Inputs
-------

I                                     Input image (ny x nx x P) in UINT8 format. 

options
        N                             Number of sampling points (1 x nR) (default [N=8])
        R                             Vector of Radius (1 x nR) (default [R=1]) ny>2max(R)+1 & nx>2max(R)+1                                     
        map                           Mapping of the chlbp (2^Nmax x nR) in double format (default map = (0 : 255))
        shiftbox                      Shifting box parameters shiftbox (2 x 2 x nR) where [by , bx ; deltay , deltax] x nR (default shiftbox = [ny , nx ; 0 , 0])
                                      by, bx denote the size of subwindows analysis, deltay, deltax represent the shift the subwindows analysis.
Outputs
-------

H                                     chlbp features (bin*nH x P) int UINT32 format where nH = nR*max(floor((ny - shiftbox(1,1,:))./shiftbox(2,1,:)) + 1 , 1)*max(floor((nx - shiftbox(1,2,:))./shiftbox(2,2,:)) + 1 , 1)

To compile
----------

mex  -output chlbp.dll chlbp.c
mex  -f mexopts_intel10.bat -output chlbp.dll chlbp.c
mex  -f mexopts_intel11.bat  chlbp.c



clear, close all 
load viola_24x24 
Ny = 24; 
Nx = 24; 
options.N = [8 , 12]; 
options.R = [2 , 2]; 
options.map = zeros(2^max(options.N) , length(options.N));

mapping = getmapping(options.N(1),'u2'); 
options.map(1:2^options.N(1) , 1) = mapping.table'; 

mapping = getmapping(options.N(2),'u2'); 
options.map(1:2^options.N(2) , 2) = mapping.table'; 
options.shiftbox = cat(3 , [Ny , Nx ; 1 , 1]  , [Ny , Nx ; 1 , 1]);

H = chlbp(X , options);
H2 = chlbp(X(:,:,2) , options);

[H2 , H(: , 2)]





P                    = 256;
I                    = (imresize(imread('rice.png') , [P , P]));
options.N            = [8 , 8];
options.R            = [1 , 2];
options.map          = zeros(2^max(options.N) , length(options.N));
for i =1:length(options.N)
 mapping             = getmapping(options.N(i),'u2');
 options.map(: , i)  = mapping.table';
end
options.shiftbox     = cat(3 , [P/4 , P/4 ; P/8 , P/8] , [P/4 , P/4 ; P/8 , P/8]);

H                    = chlbp(I , options);




Example 1  L_{8;1,2}^u
---------

clear, close all
I                  = (imread('rice.png'));
options.N          = 8;
options.R          = 1;
mapping            = getmapping(options.N,'u2');
options.map        = mapping.table';
options.shiftbox   = [64 , 64 ; 16 , 16];
H                  = chlbp(I , options);
plot(H)
title(sprintf('CHLBP Histogram with N = %d, R = %d' , options.N , options.R))



Example 2  L_{4;1}
---------
clear, close all
Ny                 = 19;
Nx                 = 19;
I                  = uint8(ceil(255*rand(Ny , Nx)));
options.N          = 4;
options.R          = 1;
options.shiftbox   = [10 , 10 ; 4 , 4];
H                  = chlbp(I , options);


Example 3  L_{8;1}
---------
clear, close all
load viola_24x24
Ny                 = 24;
Nx                 = 24;
options.N          = 8;
options.R          = 1;
mapping            = getmapping(options.N,'u2');
options.map        = mapping.table';

H                  = chlbp(X ,options);
imagesc(H)


Example 4  L_{8;1}^u + L_{4,1}
---------

clear, close all
Ny                                 = 19;
Nx                                 = 19;
X                                  = uint8(ceil(255*rand(Ny , Nx)));

options.N                          = [8 , 4];
options.R                          = [1 , 1];
map                                = zeros(2^max(options.N) , length(options.N));
mapping                            = getmapping(options.N(1),'u2');
options.map(: , 1)                 = mapping.table';
options.map(1:2^options.N(2) , 2)  = (0:2^options.N(2)-1)';
options.shiftbox                   = cat(3 , [Ny , Nx ; 0 , 0] , [10 , 10 ; 4 , 4]);


H                                   = chlbp(X , options);
plot(H)


Example 5  L_{8;1}^u + L_{4;1} + Gentleboosting
---------

clear, close all
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

options.T                          = 50;


H                                  = chlbp(X , options);
figure(1)
imagesc(H)
title('CHLBP Features')
drawnow

y                                  = int8(y);
indp                               = find(y == 1);
indn                               = find(y ==-1);

index                              = randperm(length(y)); %shuffle data to avoid numerical discrepancies with long sequence of same label

options.param                      = chlbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options);
[yest_train , fx_train]            = chlbp_gentleboost_binary_predict_cascade(H , options);

tp_train                           = sum(yest_train(indp) == y(indp))/length(indp)
fp_train                           = 1 - sum(yest_train(indn) == y(indn))/length(indn)
Perf_train                         = sum(yest_train == y)/length(y)

[dum , ind]                        = sort(y , 'descend');
figure
plot(fx_train(ind))
title('Output of the strong classifier for train data')


[tpp_train , fpp_train]            = basicroc(y , fx_train);

load jensen_24x24
H                                  = chlbp(X , options);
[yest_test , fx_test]              = chlbp_gentleboost_binary_predict_cascade(H , model);

tp_test                            = sum(yest_test(indp) == y(indp))/length(indp)
fp_test                            = 1 - sum(yest_test(indn) == y(indn))/length(indn)
Perf_test                          = sum(yest_test == y)/length(y)

[dum , ind]                        = sort(y , 'descend');
figure
plot(fx_test(ind))
title('Output of the strong classifier for test data')


[tpp_test , fpp_test]              = basicroc(y , fx_train);


figure
plot(fpp_train , tpp_train , fpp_test , tpp_test , 'r' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
title('ROC curves')
legend('Train' , 'Test')



Example 6  L_{8;1}^u + L_{4;1} + L_{12;2}^u + Gentleboosting
---------

clear, close all
load viola_24x24
Ny                                  = 24;
Nx                                  = 24;
options.N                           = [8 , 4 , 12];
options.R                           = [1 , 1 , 2];
options.map                         = zeros(2^max(options.N) , length(options.N));

mapping                             = getmapping(options.N(1),'u2');
options.map(1:2^options.N(1) , 1)   = mapping.table';
options.map(1:2^options.N(2) , 2)   = (0:2^options.N(2)-1)';
mapping                             = getmapping(options.N(3),'u2');
options.map(1:2^options.N(3) , 3)   = mapping.table';
options.shiftbox                    = cat(3 , [Ny , Nx ; 1 , 1] , [16 , 16 ; 4 , 4] , [Ny , Nx ; 1 , 1]);

options.T                           = 50;

H                                   = chlbp(X , options);
figure
imagesc(H)
title('CHLBP Features')
drawnow

y                                   = int8(y);
indp                                = find(y == 1);
indn                                = find(y ==-1);

index                               = randperm(length(y)); %shuffle data to avoid numerical discrepancies with long sequence of same label

options.model                       = chlbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options);
[yest_train , fx_train]             = chlbp_gentleboost_binary_predict_cascade(H , options);

tp_train                            = sum(yest_train(indp) == y(indp))/length(indp)
fp_train                            = 1 - sum(yest_train(indn) == y(indn))/length(indn)
Perf_train                          = sum(yest_train == y)/length(y)

[dum , ind]                         = sort(y , 'descend');
figure
plot(fx_train(ind))
title('Output of the strong classifier for train data')


[tpp_train , fpp_train]             = basicroc(y , fx_train);

load jensen_24x24
H                                   = chlbp(X , options);
[yest_test , fx_test]               = chlbp_gentleboost_binary_predict_cascade(H , options);

tp_test                             = sum(yest_test(indp) == y(indp))/length(indp)
fp_test                             = 1 - sum(yest_test(indn) == y(indn))/length(indn)
Perf_test                           = sum(yest_test == y)/length(y)

[dum , ind]                         = sort(y , 'descend');
figure
plot(fx_test(ind))
title('Output of the strong classifier for test data')


[tpp_test , fpp_test]               = basicroc(y , fx_train);


figure
plot(fpp_train , tpp_train , fpp_test , tpp_test , 'r' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
title('ROC curves')
legend('Train' , 'Test')



Author : Sébastien PARIS : sebastien.paris@lsis.org
-------  Date : 01/20/2009

Reference ""


*/


#include <math.h>
#include <mex.h>

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif

#define round(f)  ((f>=0)?(int)(f + .5):(int)(f - .5)) 

#define M_PI 3.14159265358979323846
#define huge 1e300
#define tiny 1e-6
#define NOBIN -1
#define intmax 32767 

struct opts
{
	double        *R;
	int           nR;
	double        *N;
	double        *shiftbox;
	double        *map;
	int           *bin;
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void qs( double * , int , int  ); 
int	number_chlbp_features(int , int , int * , double * , int );
void chlbp(unsigned char * , int  , int  , int , struct opts , unsigned int * , int  );

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *I;
	int *dimsH;
	const int *dimsI , *dimsshiftbox;
	int numdimsI , numdimsshiftbox;
	unsigned int *H;
	int bincurrent;
	int i , j , ny , nx , P=1 , maxN=-1 , nH = 1 , powmaxN , powN  = 256 , indmaxN , indnR;
	struct opts options;
	double *mapsorted;
	double  currentbin , maxR ;
	mxArray *mxtemp;



	options.nR  = 1;


	/* Input 1  */

	numdimsI    = mxGetNumberOfDimensions(prhs[0]);


	if( (numdimsI > 3) || !mxIsUint8(prhs[0]) )
	{
		mexErrMsgTxt("I must be (ny x nx x P) in double format");
	}


	I           = (unsigned char *)mxGetData(prhs[0]);
	dimsI       = mxGetDimensions(prhs[0]);
	ny          = dimsI[0];
	nx          = dimsI[1];
	if(numdimsI == 3)
	{
		P      = dimsI[2];
	}

	/* Input 2  */

    if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
    {
		mxtemp     = mxGetField(prhs[1] , 0 , "N");

		if(mxtemp != NULL)
		{
			options.N        = mxGetPr(mxtemp);
			options.nR       = mxGetN(mxtemp);
			for (i = 0 ; i < options.nR ; i++)
			{
				maxN  = max(maxN , (int)options.N[i]);
			}
			powmaxN    = (int) (pow(2 , maxN));
		}
		else
		{
			options.N        = (double *)mxMalloc(sizeof(double));
			options.N[0]     = 8;
			options.nR       = 1;
		}

		mxtemp               = mxGetField(prhs[1] , 0 , "R");
		if(mxtemp != NULL)
		{
			if( (mxGetNumberOfDimensions(mxtemp) != 2) || (mxGetN(mxtemp) != options.nR) )
			{
				mexErrMsgTxt("R must be (1 x nR)");
			}
			options.R          = mxGetPr(mxtemp);
			maxR               = options.R[0];
			for(i = 1 ; i < options.nR ; i++)
			{
				maxR   = max(maxR , options.R[i]);
			}
		}
		else
		{
			options.R              = (double *)mxMalloc(sizeof(double));
			options.R[0]           = 1.0;
			options.nR             = 1;
		}

		mxtemp               = mxGetField(prhs[1] , 0 , "map");
		if(mxtemp != NULL)
		{
			if( (mxGetNumberOfDimensions(mxtemp) != 2) || mxGetM(mxtemp) != powmaxN || mxGetN(mxtemp) != options.nR )
			{
				mexErrMsgTxt("map must be (2^Nmax x nR)");
			}

			options.map              = mxGetPr(mxtemp);

			/* Determine unique values in map vector */

			mapsorted                = (double *)mxMalloc(powmaxN*sizeof(double));
			options.bin              = (int *) mxMalloc(options.nR*sizeof(int));
			indmaxN                  = 0;

			for(j = 0 ; j < options.nR ; j++)
			{
				powN = (int) (pow(2 , (int) options.N[j]));

				for ( i = 0 ; i < powN ; i++ )
				{
					mapsorted[i] = options.map[i + indmaxN];
				}

				qs( mapsorted , 0 , powN - 1 );
				bincurrent    = 0;
				currentbin    = mapsorted[0];

				for (i = 1 ; i < powN ; i++)
				{
					if (currentbin != mapsorted[i])
					{
						currentbin = mapsorted[i];
						bincurrent++;
					}
				}
				bincurrent++;
				options.bin[j]  = bincurrent;
				indmaxN        += powmaxN;						
			}
		}
		else
		{
			options.map           = (double *)mxMalloc(powmaxN*options.nR*sizeof(double));
			options.bin           = (int *) mxMalloc(options.nR*sizeof(int));
			indmaxN               = 0;

			for(j = 0 ; j < options.nR ; j++)
			{
				powN = (int) (pow(2 , (int) options.N[j]));
				for(i = indmaxN ; i < powN + indmaxN ; i++)
				{
					options.map[i] = i;
				}
				indmaxN           += powmaxN;
				options.bin[j]    = powN;
			}
		}

		mxtemp               = mxGetField(prhs[1] , 0 , "shiftbox");
		if(mxtemp != NULL)
		{
			if( mxGetNumberOfDimensions(mxtemp) > 3 )
			{
				mexErrMsgTxt("shiftbox must be (2 x 2 x nbox)");
			}
			numdimsshiftbox          = mxGetNumberOfDimensions(mxtemp);
			dimsshiftbox             = mxGetDimensions(mxtemp);

			if(dimsshiftbox[0] !=2 || dimsshiftbox[1] !=2 )
			{
				mexErrMsgTxt("shiftbox must be (2 x 2 x nR)");
			}
			options.shiftbox         = mxGetPr(mxtemp);
			if(numdimsshiftbox == 3)
			{
				if(dimsshiftbox[2] != options.nR)
				{
					mexErrMsgTxt("shiftbox must be (2 x 2 x nR)");
				}
			}
			for(i = 0 ; i < options.nR ; i++)
			{
				if( (options.shiftbox[0 + i*4] < 2.0*maxR+1) || (options.shiftbox[2 + i*4] < 2.0*maxR+1) )
				{
					mexErrMsgTxt("by and bx must be > 2*max(R) + 1");
				}
			}   
		}
		else
		{
			options.shiftbox          = (double *)mxMalloc(4*options.nR*sizeof(double));
			indnR                     = 0;

			for( i = 0 ; i < options.nR ; i++)
			{
				options.shiftbox[0 + indnR]       = (double) ny;
				options.shiftbox[1 + indnR]       = 0.0;
				options.shiftbox[2 + indnR]       = (double) nx;
				options.shiftbox[3 + indnR]       = 0.0;
				indnR                            += 4;
			}      
		}
	}

	nH                 = number_chlbp_features(ny , nx , options.bin , options.shiftbox , options.nR );

	/*----------------------- Outputs -------------------------------*/

	dimsH              = (int *)mxMalloc(2*sizeof(int));
	dimsH[0]           = nH;
	dimsH[1]           = P;
	plhs[0]            = mxCreateNumericArray(2 , dimsH , mxUINT32_CLASS , mxREAL);
	H                  = (unsigned int *)mxGetPr(plhs[0]);


	/*------------------------ Main Call ----------------------------*/

	chlbp(I , ny , nx , P , options , H  , nH );

	/*--------------------------- Free memory -----------------------*/

	mxFree(dimsH);
	mxFree(options.bin);


	if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{
		if ( (mxGetField( prhs[1] , 0 , "N" )) == NULL )
		{
			mxFree(options.N);
		}
		if ( (mxGetField( prhs[1] , 0 , "R" )) == NULL )
		{
			mxFree(options.R);
		}
		if ( (mxGetField( prhs[1] , 0 , "map" )) == NULL )
		{
			mxFree(options.map);
		}
		else
		{
			mxFree(mapsorted);
		}
		if ( (mxGetField( prhs[1] , 0 , "shiftbox" )) == NULL )
		{
			mxFree(options.shiftbox);
		}
	}
	else
	{
		mxFree(options.N);
		mxFree(options.R);
		mxFree(options.map);
		mxFree(options.shiftbox);
	}
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */
void chlbp(unsigned char *I , int ny , int nx , int P , struct opts options , unsigned int *H  , int nH )
{
	double *N = options.N , *R = options.R , *shiftbox = options.shiftbox, *map = options.map;
	int *bin = options.bin;
	int nR = options.nR;
	double a , radius , minx , miny , maxx , maxy, temp;
	double x , y , tx  , ty ;
	double w1 , w2 , w3 , w4;
	int by , deltay , bx , deltax , bymax=-1 , bxmax=-1;
	int bsizey , bsizex , dy , dx , floory , floorx , origy , origx , minR=intmax , nynx = ny*nx , indnynx;
	int ly , lx , offsety , offsetx , indCx , indrx , indfx , indcx , dimDy , dimDx , dimD , indD , indbox;
	int rx , ry , fx , cx  , fy , cy;
	int nbin , bincurrent , maxN = -1 , Ncurrent , indmaxN , indnH , powmaxN;
	int i , j , l , m , n , r  , c , co , k;
	double *spoints ;
	int *vectpow2 , *D;

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


	dimDy      = (bymax - minR);
	dimDx      = (bxmax - minR);
	dimD       = dimDy*dimDx;
	D          = (int *) malloc(dimD*sizeof(int)); /* (dy+1 x dx+1) */

	powmaxN    = (int) pow(2 , maxN);

	indnynx    = 0;
	indnH      = 0;

	for (c = 0 ; c < P ; c++)
	{
		indbox     = 0;
		indmaxN    = 0;
		co         = indnH;

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

			minx       = huge;
			miny       = huge;
			maxx       = -huge;
			maxy       = -huge;

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
			indmaxN    += powmaxN;
			indbox     += 4;
		}
		indnynx    += nynx;
		indnH      += nH;
	}
	free(spoints);
	free(vectpow2);
	free(D);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
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
/*----------------------------------------------------------------------------------------------------------------------------------------- */