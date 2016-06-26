/*

mex -f mexopts_intel10.bat -output fast_rotate.dll fast_rotate.c

mex -v -DOMP -f mexopts_intel10.bat -output fast_rotate.dll fast_rotate.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"

clear, close all
rgb   = imread('class57.jpg');
I     = rgb2gray(rgb);
Irot  = fast_rotate(I , -12);

figure(1)
imagesc(I)
colormap(gray)

figure(2)
imagesc(Irot)
colormap(gray)
*/

#include <mex.h>
#include <math.h>

const double PI = 3.14159265358979323846;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	unsigned char * source;
	unsigned char * dest;
	const int *dims;
	int num_of_dims;
	float angle, rad , ca , sa , ux , uy , vx , vy , w2 , h2;
	int X,Y; 
	int x,y; 
	int index_Height;
	int width, height;

	if (nrhs != 2)
	{
		mexErrMsgTxt("Usage : fast_rotate(image,ang)");
	}

	/* Input 1 */
	source      = (unsigned char *)mxGetData(prhs[0]);
	dims        = mxGetDimensions(prhs[0]);
	num_of_dims = mxGetNumberOfDimensions(prhs[0]);
	width       = dims[0];
	height      = dims[1];

	/* Input 2 */

	angle        = (float)mxGetScalar(prhs[1]);

	/* Output */

	plhs[0]      = mxCreateNumericArray(num_of_dims, dims, mxUINT8_CLASS, mxREAL);
	dest         = (unsigned char *)mxGetData(plhs[0]);

	rad          = (float)((angle*PI)/180.0);
	ca           = (float)cos(rad);
	sa           = (float)sin(rad);

	ux           = (float)(abs(width*ca));
	uy           = (float)(abs(width*sa));
	vx           = (float)(abs(height*sa));
	vy           = (float)(abs(height*ca));
	w2           = 0.5f*width;
	h2           = 0.5f*height;

#ifdef OMP 
#pragma omp parallel for default(none) private(X,Y,y,x,index_Height) shared(source,dest,w2,h2,ca,sa,width,height) 
#endif
	for(y = 0 ; y < height ; y++)
	{
		index_Height = y*width;
		for(x=0 ; x<width ; x++)
		{
			X                      = (int)(w2 + (x-w2)*ca + (y-h2)*sa+0.5); 
			Y                      = (int)(h2 - (x-w2)*sa + (y-h2)*ca+0.5); 
			dest[x + index_Height] = (X<0 || Y<0 || X>=width || Y>=height)?0:source[X + Y*width];
		}
	}
}
