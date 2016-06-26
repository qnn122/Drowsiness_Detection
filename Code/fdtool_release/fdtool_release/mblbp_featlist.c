
/*

  MultiBlock Local Binary Pattern features list parameters

  Usage
  ------

  F = mblbp_featlist([ny] , [nx] , [scale]);

  
  Inputs
  -------

  ny                                    Number of rows of the pattern (default ny = 24)
  nx                                    Number of columns of the pattern (default nx = ny)
  scale                                 Scaling box matrix (2 x Nscale) where scale = [w(1) , w(2) , ... w(Nscale) ; h(1) , h(2) , .... , h(Nscale)]
 

  Outputs
  -------
  
  F                                     Features lists (5 x nF) int UINT32 where nF design the total number of mblbp features 
                                        F(: , i) = [if ; xf ; yf ; wf ; hf] where
									    if       index of the current feature, if = [1,...,nF]
									    xf,yf    coordinates of the current feature (top-left rectangle)
                                        wf,hf    width and height of each of the 9 rectangles

  To compile
  ----------


  mex  -g -output mblbp_featlist.dll mblbp_featlist.c

  mex  -f mexopts_intel10.bat -output mblbp_featlist.dll mblbp_featlist.c


  Example 1
  ---------

  F         = mblbp_featlist(3);

  Example 2
  ---------

  F         = mblbp_featlist(20 , 20);


  Example 3
  ---------

  F         = mblbp_featlist(24 , 24 , [1 , 3 ; 1 , 3]);


  Example 4
  ---------

  F         = mblbp_featlist(20 , 20 , [1 , 2 , 3 , 4 , 5 , 6 ; 1 , 2 , 3 , 4 , 5 , 6]);


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/20/2009

 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999

              [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
			      ICB07

			  [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
			  [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004

*/


#include <math.h>
#include <mex.h>

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

int number_mblbp_features(int , int , double * , int );
void mblbp_featlist(int  , int , double * , int  , unsigned int *);

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    int ny = 24 , nx = 24;
	double *scale;
	int Nscale = 0;
	int nF;
	unsigned int *F;
	int *dimsF;

    /* Input 1  */    
    if ((nrhs > 0) && (int)mxIsScalar(prhs[0]) )
    {        
        ny        = (int) mxGetScalar(prhs[0]);
    }
	if(ny < 3)
	{
		mexErrMsgTxt("ny must be >= 3");
	}
    
    /* Input 2  */
    
    if ((nrhs > 1) && (int)mxIsScalar(prhs[1]) )    
    {        
        nx        = (int) mxGetScalar(prhs[1]);
		if(nx < 3)
		{
			mexErrMsgTxt("nx must be >= 3");		
		}
    }
	else
	{	
		nx        = ny;	
	}
    if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
	{
		if(mxGetM(prhs[2]) !=2)
		{		
			mexErrMsgTxt("scale must be a (2 x Nscale) matrix");	
		}
		scale     = mxGetPr(prhs[2]);
		Nscale    = mxGetN(prhs[2]);
	}

	nF            = number_mblbp_features(ny , nx , scale , Nscale);

    /*------------------------ Output ----------------------------*/

	dimsF         = (int *)mxMalloc(2*sizeof(int));
	dimsF[0]      = 5;
	dimsF[1]      = nF;
	plhs[0]       = mxCreateNumericArray(2 , dimsF , mxUINT32_CLASS , mxREAL);
	F             = (unsigned int *)mxGetPr(plhs[0]);

    /*------------------------ Main Call ----------------------------*/
      
    mblbp_featlist(ny , nx , scale , Nscale, F);
    
	/*----------------- Free Memory --------------------------------*/

	mxFree(dimsF);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void mblbp_featlist(int ny , int nx , double *scale , int Nscale , unsigned int *F)
{
	int i , j , w = 1 , h , nofeat = 1 , co = 0 , s , inds; 
	if(Nscale == 0)
	{	
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
	else
	{	
		inds   = 0;

		for(s = 0 ; s < Nscale ; s++)
		{
			w     = (int) scale[0 + inds];
			h     = (int) scale[1 + inds];

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
			inds += 2;
			nofeat++;		
		}
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int number_mblbp_features(int ny , int nx , double *scale , int Nscale)
{
	int nF = 0 , w=1 , h , nx1 = nx + 1 , ny1 = ny + 1 , temp , inds , s;

	if(Nscale == 0)
	{	
		while(nx >= 3*w)
		{		
			temp = (nx1 - 3*w);	
			h    = 1;
			
			while(ny >= 3*h)
			{
				nF  += (ny1 - 3*h)*temp;
				h++;
			}
			w++;
		}
	}
	else
	{
		inds  = 0;
		for (s = 0 ; s < Nscale ; s++)
		{
			w     = (int) scale[0 + inds];
			h     = (int) scale[1 + inds];
			nF   += (nx1 - 3*w)*(ny1 - 3*h);
			inds += 2;
		}
	}

	return nF;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
