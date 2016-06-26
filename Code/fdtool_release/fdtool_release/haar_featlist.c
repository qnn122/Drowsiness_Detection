
/*

  Haar features list parameters 

  Give all Haar features parameters for a Image I (ny x nx) and dictionary of Haar pattern given
  by rect_param

  Usage
  ------

  F = haar_featlist([ny] , [nx] , [rect_param]);

  
  Inputs
  ------

  ny                                    Number of rows of the pattern (default ny = 24)
  nx                                    Number of columns of the pattern (default nx = ny)
  rect_param                            Features rectangles parameters (10 x nR), where nR is the total number of rectangles for the patterns.
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

										Please run gui_features_dictionary in the \gui subdir to build such parameters

  Output
  ------
  
  F                                     Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
                                        F(: , i) = [if ; xf ; yf ; wf ; hf ; ir]
										if     index of the current feature, if = [1,....,nF] where nF is the total number of Haar's features  (see nbfeat_haar function)
										xf,yf  top-left coordinates of the current Haar's feature
										wf,hf  width and height of the current feature Haar's feature
										ir     Linear index of the FIRST rectangle of the current Haar's feature according rect_param definition. ir is used internally in Haar function
										       (ir/10 + 1) is the matlab index of this first rectangle
  To compile
  ----------


  mex  -g -output haar_featlist.dll haar_featlist.c

  mex  -f mexopts_intel10.bat -output haar_featlist.dll haar_featlist.c

  Example 1
  ---------


  F         = haar_featlist(3);


  Example 2
  ---------

  F         = haar_featlist(24 , 24);


  Example 3
  ---------

  load haar_dico_2.mat;
  F         = haar_featlist(24 , 24 , rect_param);


  Example 4
  ---------

  load haar_rect_param
  F         = haar_featlist(3 , 3 , rect_param(: , 1:2));




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

int number_haar_features(int , int , double * , int );
void haar_featlist(int  , int , double * , int  , unsigned int * );
/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    int ny = 24 , nx = 24;
	double	rect_param_default[40] = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	double *rect_param;
	int nR = 4;
	int i , nF;
	unsigned int *F;
	int *dimsF;

    /* Input 1  */

    if ((nrhs > 0) && mxIsScalar(prhs[0]) )  
    {        
        ny        = (int) mxGetScalar(prhs[0]);        
    }
	if(ny < 3)
	{
		mexErrMsgTxt("ny must be >= 3");
	}
    
    /* Input 2  */
    
    if ((nrhs > 1) && mxIsScalar(prhs[1]) )
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
	
    if ((nrhs > 2) && !mxIsEmpty(prhs[2])  )
	{
		if(mxGetM(prhs[2]) != 10)
		{		
			mexErrMsgTxt("rect_param must be a (10 x nR) matrix");	
		}
		rect_param     = mxGetPr(prhs[2]);
		nR             = mxGetN(prhs[2]);
	}
	else
	{
		rect_param     = (double *)mxMalloc(40*sizeof(double));
		for (i = 0 ; i < 40 ; i++)
		{
			rect_param[i]    = rect_param_default[i];
		}			
	}

	nF            = number_haar_features(ny , nx , rect_param , nR);


	dimsF         = (int *)mxMalloc(2*sizeof(int));
	dimsF[0]      = 6;
	dimsF[1]      = nF;
	plhs[0]       = mxCreateNumericArray(2 , dimsF , mxUINT32_CLASS , mxREAL);
	F             = (unsigned int *)mxGetPr(plhs[0]);

    /*------------------------ Main Call ----------------------------*/
        
	haar_featlist(ny , nx , rect_param , nR , F );

    /*------------------------ Free memory ----------------------------*/

	mxFree(dimsF);

    if ( (nrhs < 3) || mxIsEmpty(prhs[2]) )
	{
		mxFree(rect_param);
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
