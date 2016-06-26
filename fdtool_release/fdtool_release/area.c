
/*

  A        = area(I , y , x , h , w);

  To compile
  ----------

  mex  -g -output area.dll area.c

  mex  -f mexopts_intel10.bat -output area.dll area.c

  Example 1
  ---------

  clear, close all
  load viola_24x24

  I         = X(: , : , 1);
  II        = cumsum(cumsum(double(I) , 1) , 2);


  y         = 2;
  x         = 4;
  h         = 10;
  w         = 11;
  A1        = area(I , y , x , h , w);
  A2        = sum(sum(I(y:y+h-1,x:x+w-1)));
 % A2        = II(y+h,x+w) + II(y,x) - (II(y+h,x)+II(y,x+w));


*/

#include <math.h>
#include <mex.h>

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

void MakeIntegralImage(unsigned char *, unsigned int  *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *I;
	const int *dimsI;
	int numdimsI;
	unsigned int *II , *Itemp;
	int y , x , h , w;
	double ai;
	double *A;
    int Ny , Nx , NyNx , N = 1;

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

	NyNx                       = Ny*Nx;

	y                          = mxGetScalar(prhs[1]);
	x                          = mxGetScalar(prhs[2]);
	h                          = mxGetScalar(prhs[3]);
	w                          = mxGetScalar(prhs[4]);

    
	II                         = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Itemp                      = (unsigned int *) malloc(NyNx*sizeof(unsigned int));

	MakeIntegralImage(I , II , Nx , Ny , Itemp);

	ai                          = (double)Area(II , x - 1 , y - 1 , w , h , Ny);


    plhs[0]                    = mxCreateDoubleMatrix(1 , 1 , mxREAL);
    A                          = mxGetPr(plhs[0]);

	A[0]                       = ai;

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
