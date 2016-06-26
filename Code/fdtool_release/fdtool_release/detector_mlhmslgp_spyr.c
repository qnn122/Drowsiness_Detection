/*

  Object detector based on fast Histogram of MBLGP features through Histogram Integral and trained by Linear SVM 

  Usage
  -----

  [D , stat , [matfx]] = detector_mlhmslgp_spyr(I , [model]);

  
  Inputs
  ------

  I                                     Input image (Ny x Nx) in UINT8 format
  
  model                                 Trained model structure

             w                          Trained model with a Linear SVM, weight vector (1 x ((1+improvedLGP)*Nbins*nH*nscale+addbias)).
			                            where Nbins = ([256,59,36,10]*(improvedLGP+1)) if cs_opt = 0, Nbins = ([16,15,10,10]*(improvedLGP+1)) if cs_opt = 1.
			 addbias                    Add bias or not for model prediction (1/0).
             homtable                   Precomputed table for homogeneous additive Kernel approximation (used when model.n > 0).
			 n                          Order approximation for the homogeneous additive Kernel.
			 L                          Sampling step (default L = 0.5)
			 kerneltype                 0 for intersection kernel, 1 for Jensen-shannon kernel, 2 for Chi2 kernel (default kerneltype = 0).
		     numsubdiv                  Number of subdivisions (default numsubdiv = 8).
             minexponent                Minimum exponent value (default minexponent = -20).
             maxexponent                Maximum exponent value (default minexponent = 8).
             spyr                       Spatial Pyramid (nspyr x 5) (default [1 , 1 , 1 , 1 , 1] with nspyr = 1)
                                        where spyr(i,1) is the ratio of ny in y axis of the blocks at level i (by(i) = spyr(i,1)*ny),
                                        where spyr(i,2) is the ratio of nx in x axis of the blocks at level i (bx(i) = spyr(i,3)*nx),
                                        where spyr(i,3) is the ratio of ny in y axis of the shifting at level i (deltay(i) = spyr(i,2)*ny),
                                        where spyr(i,4) is the ratio of nx in x axis of the shifting at level i (deltax(i) = spyr(i,4)*nx),
                                        where spyr(i,5) is the weight's histogram associated to current level pyramid (w(i) = spyr(i,1)*spyr(i,2))
										total number of subwindows nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1))
			 nH                         Number of subwindows associated with spyr (default nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1)))
             scale                      Multi-Scale vector (1 x nscale) (default scale = 1) where scale(i) = s is the size's factor to apply to each 9 blocks
                                        in the LGP computation, i = 1,...,nscale
			 cs_opt                     Center-Symetric LGP : 1 for computing CS-MBLBP features, 0 : for MBLGP (default cs_opt = 0)
             improvedLGP                0 for default 8/4 bits encoding, 1 for 9/5 bits encoding (8/4 bits + dirac{8/4*central area>sum(Ai)})
 	         norm                       Normalization vector (1 x 3) : [for all subwindows, for each subwindows of a pyramid level, for each subwindows]
                                        norm = 0 <=> no normalization, norm = 1 <=> v=v/(sum(v)+epsi), norm = 2 <=> v=v/sqrt(sum(v²)+epsi²),
	                                    norm = 3 <=> v=sqrt(v/(sum(v)+epsi)) , norm = 4 <=> L2-clamped (default norm = [0 , 0 , 4])
			 clamp                      Clamping value (default clamp = 0.2)
	         maptable                   Mapping table for LGP codes. LGP code belongs to {0,...,b}, where b is defined according to following table:
								        |maptable | cs_opt = 0, improvedLGP = 0 | cs_opt = 0, improvedLGP = 1 | cs_opt = 1, improvedLGP = 0 | cs_opt = 1, improvedLGP = 1|
										|   0     |           255               |              511            |            15               |              31            |
										|   1     |           58                |              117            |            14               |              29            |
										|   2     |           35                |              71             |            5                |              11            |
										|   3     |           9                 |              19             |            5                |              11            |
             rmextremebins              Force to zero bin = {0 , b} if  rmextremebins = 1 where b is defined in previous tab (default rmextremebins = 1)
             postprocessing             Type of postprocessing in order to reduce false alarms (default postprocessing = 1): 
			                            0: no postprocessing, i.e. raw detections, 1: merging if rectangles overlapp more than 25%
										2 : Better Merging detections algorithm with parameters defined by mergingbox
		     dimsIscan                  Initial Size of the scanning windows, i.e. (ny x nx ) (default dimsIscan = [24 , 24])
             scalingbox                 [scale_ini , scale_inc , step_ini] where :
                                         scale_ini is starting scale factor for each subwindows to apply from the size of trained images (default scale_ini = 2)
                                         scale_inc is Increment of the scale factor (default scale_inc = 1.4)
					                     step_ini  is the overlapping subwindows factor such that delta = Round(step_ini*scale_ini*scale_inc^(s)) where s in the number of scaling steps (default step_ini = 2)
			 mergingbox                 [overlap_same , overlap_diff , step_ini]
                                        overlap_same is the overlapping factor for merging detections of the same size (first step) (default overlap_same = 1/2)
                                        overlap_diff is the overlapping factor for merging detections of the different size (second step) (default overlap_diff = 1/2)
					                    dist_ini is the size fraction of the current windows allowed to merge included subwindows (default dist_ini = 1/3)
			 max_detections             Maximum number of raw subwindows detections (default max_detections = 500)

If compiled with the "OMP" compilation flag

			 num_threads                Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  D                                     Detection result (5 x nD) where nD is the number of detection
                                        D(1,:) x coordinates of the detections
                                        D(2,:) y coordinates of the detections
                                        D(3,:) size of detection windows
										D(4,:) number of merged detection
										D(5,:) detection'values

  stat                                  Number of positives and negatives detection of all scanned subwindows(1 x 2)

  If compiled with the "matfx" compilation flag

  matfx                                 Matrix of raw detections (Ny x Nx)


  To compile
  ----------

  mex  -g detector_mlhmslgp_spyr.c

  mex  detector_mlhmslgp_spyr.c

  mex  -f mexopts_intel10.bat detector_mlhmslgp_spyr.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat detector_mlhmslgp_spyr.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"

  or with the matfx option

  mex  -Dmatfx -f mexopts_intel10.bat detector_mlhmslgp_spyr.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -Dmatfx -f mexopts_intel10.bat detector_mlhmslgp_spyr.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"


  Example 1
  ---------

  clear,close all

  model                    = load('modelw8.mat');
 % model.spyr               = [1 , 1 , 1 , 1 ; 1/2 , 1/2 , 1/4 , 1/4];
  model.spyr                = [1 , 1  , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4];

  model.nH                 = sum(floor(((1 - model.spyr(:,1))./(model.spyr(:,3)) + 1)).*floor((1 - model.spyr(:,2))./(model.spyr(:,4)) + 1));
  model.scale              = [1];
  model.dimsIscan          = [24 , 24];
  model.maptable           = 1;
  model.cs_opt             = 0;
  model.improvedLGP        = 0;
  model.rmextremebins      = 1;
  model.norm               = [0 , 0 , 2];
  model.clamp              = 0.2;
  model.n                  = 0;
  model.L                  = 1;
  model.kerneltype         = 0;
  model.numsubdiv          = 8;
  model.minexponent        = -20;
  model.maxexponent        = 8;
  model.scalingbox         = [2 , 1.4 , 1.8];
  model.mergingbox         = [1/2 , 1/2 , 0.8];
  model.postprocessing     = 0;
  model.max_detections     = 10000;
  model.num_threads        = -1;
  model.addbias            = 1;

% if(model.maptable == 0)
%     model.Nbins           = 256;
% elseif(model.maptable == 1)
%     model.Nbins           = 59;
% elseif(model.maptable == 2)
%     model.Nbins           = 36;
% elseif(model.maptable == 3)
%     model.Nbins           = 10;
% end
% model.w                  = randn(1,model.Nbins*length(model.scale)*model.nH+model.addbias);

 min_detect               = 110;
  
 load Itest
 
 [D , stat]               = detector_mlhmslgp_spyr(I , model);

 rect                     = [D(1 , :)' , D(2 , :)' , D(3 , :)' , D(3 , :)'];
 [z,its]                  = size(D);

  cte = 0;
   for i = 1:size(D,2)
  %  [fx(i) , yfx  , H ]  = eval_hmblgp_spyr_subwindow(I(D(2,i):D(2,i)+D(3,i)-1 , D(1,i):D(1,i)+D(3,i)-1) , model);
	[fx(i) , yfx  , H ]  = eval_hmblgp_spyr_subwindow(I(D(2,i)-(1-cte)*1:D(2,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1 , D(1,i)-(1-cte)*1:D(1,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1) , model);

  end

  indfx                 = find(fx > 0);

  %close all
  figure(1), imshow(I);
  hold on;

  for i=1:its
    if(D(4,i)<min_detect)
        rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[1,0,0],'LineWidth',2);
    else
        rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    end
	if(model.postprocessing>0)
     text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);
	end
  end
  title(sprintf('Detect = %5.4f%%, Non-Detect = %5.4f%%'  , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))

  figure(2), imshow(I);
  hold on;
  for i=1:its
  if(D(4,i)>=min_detect)
    rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);

  end
  end
  title(sprintf('Detect = %5.4f%%, Non-Detect = %5.4f%%' , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))


  figure(3), imshow(I);
  hold on;

  for i=indfx
    if(D(4,i)<min_detect)
        rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[1,0,0],'LineWidth',2);
    else
        rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    end
	if(model.postprocessing>0)
     text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);
	end
  end
  title(sprintf('Detect = %5.4f%%, Non-Detect = %5.4f%%'  , 100*length(indfx)/sum(stat) , 100*(sum(stat)-length(indfx))/sum(stat)))

  figure(4), imshow(I);
  hold on;
  for i=1:its
  if(D(4,i)>=min_detect)
    rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);

  end
  end
  title(sprintf('Detect = %5.4f%%, Non-Detect = %5.4f%%' , 100*length(indfx)/sum(stat) , 100*(sum(stat)-length(indfx))/sum(stat)))



  figure(5)
  plot(1:size(D,2),D(5 , :) , 1:size(D,2),fx , 'r')



  Example 2  In 320w200 working almost well
  ---------

  clear,close all

  model                    = load('modelw8.mat');
 % model.spyr               = [1 , 1 , 1 , 1 ; 1/2 , 1/2 , 1/2 , 1/2];
  model.spyr                = [1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 ];

  model.nH                 = sum(floor(((1 - model.spyr(:,1))./(model.spyr(:,3)) + 1)).*floor((1 - model.spyr(:,2))./(model.spyr(:,4)) + 1));
  model.scale              = [1];
  model.dimsIscan          = [24 , 24];
  model.maptable           = 1;
  model.cs_opt             = 0;
  model.improvedLGP        = 0;
  model.rmextremebins      = 1;
  model.norm               = [0 , 0 , 2];
  model.clamp              = 0.2;
  model.n                  = 0;
  model.L                  = 1;
  model.kerneltype         = 0;
  model.numsubdiv          = 8;
  model.minexponent        = -20;
  model.maxexponent        = 8;
  model.scalingbox         = [2 , 1.4 , 1.8];
  model.mergingbox         = [1/2 , 1/2 , 0.8];
  model.postprocessing     = 1;
  model.max_detections     = 10000;
  model.addbias            = 1;


  aa                       = vcapg2(0,2);

  min_detect               = 12;%120;


  figure(1);set(1,'doublebuffer','on');
  while(1)
        t1   = tic;        

        aa   = vcapg2(0,0);
        pos  = detector_mlhmslgp_spyr(rgb2gray(aa) , model);

        image(aa);
        hold on
        h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g' );
        hold off
        t2   = toc(t1);
        title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));

    drawnow;
  end





  Example 3  In 320w200 working almost well
  ---------

  clear,close all

  model                    = load('modelw9.mat');
 % model.spyr               = [1 , 1 , 1 , 1 1 ; 1/2 , 1/2 , 1/2 , 1/2 , 1/4];
  model.spyr                = [1 , 1 , 1 , 1  , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];

  model.nH                 = sum(floor(((1 - model.spyr(:,1))./(model.spyr(:,3)) + 1)).*floor((1 - model.spyr(:,2))./(model.spyr(:,4)) + 1));
  model.scale              = [1];
  model.dimsIscan          = [24 , 24];
  model.maptable           = 0;
  model.cs_opt             = 1;
  model.improvedLGP        = 0;
  model.rmextremebins      = 1;
  model.norm               = [0 , 0 , 2];
  model.clamp              = 0.2;
  model.n                  = 0;
  model.L                  = 1;
  model.kerneltype         = 0;
  model.numsubdiv          = 8;
  model.minexponent        = -20;
  model.maxexponent        = 8;
  model.scalingbox         = [2 , 1.4 , 1.8];
  model.mergingbox         = [1/2 , 1/2 , 0.8];
  model.postprocessing     = 1;
  model.max_detections     = 10000;
  model.addbias            = 1;


  aa                       = vcapg2(0,2);

  min_detect               = 30;%120;


  figure(1);set(1,'doublebuffer','on');
  while(1)
        t1   = tic;        

        aa   = vcapg2(0,0);
        pos  = detector_mlhmslgp_spyr(rgb2gray(aa) , model);

        image(aa);
        hold on
        h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g' );
        hold off
        t2   = toc(t1);
        title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));

    drawnow;
  end



  Example 4  
  ---------

  clear,close all
  load model_hmblbp_R4
  min_detect               = 2;%120;
  elite                    = 0.1;
  overlap                  = 0.1;


  aa                       = vcapg2(0,3);
  figure(1);set(1,'doublebuffer','on');
  while(1)
      t1   = tic;
      
      aa   = vcapg2(0,0);
      pos  = detector_mlhmslgp_spyr(rgb2gray(aa) , model);
	  ind  = nms(pos , overlap);

%	  [valmax , indmax ] = sort(pos(5 , :) , 'descend');
%	  ind                = indmax(1:ceil(elite*length(indmax)));
      
      image(aa);
      hold on
 %     h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g' );
      h    = plot_rectangle(pos(: , ind) , 'g' );

      hold off
      t2   = toc(t1);
      title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));
      drawnow;
  end



  Example 5  
  ---------
 clear,close all

 load Itest
 load model_hmblbp_R4_H16_sp2

 model.postprocessing          = 0;           

 np                           = size(model.spyr , 1);
 maxfactor                    = max(model.spyr(: , 1).*model.spyr(: , 2));



 [D , stat]                    = detector_mlhmslgp_spyr(I , model);

 rect                          = [D(1 , :)' , D(2 , :)' , D(3 , :)' , D(3 , :)'];
 [z,its]                       = size(D);

 [fxI , yfxI  , HI , IIR , R ] = eval_hmblgp_spyr_subwindow(I , model);

 H                             = zeros(size(HI,1) , size(D,2)*model.nH);
 H1                            = zeros(size(HI,1) , size(D,2)*model.nH);
 H2                            = zeros(size(HI,1) , size(D,2)*model.nH);

 cte                           = 0;
 for i = 1:size(D,2)
    [fx(i) , yfx  , H(: , i) ]  = eval_hmblgp_spyr_subwindow(I(D(2,i)-(1-cte)*1:D(2,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1 , D(1,i)-(1-cte)*1:D(1,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1) , model);
 end



for i = 1:size(D,2)
    co                          = 1; 
    for p = 1 : np
        
        scaley      = model.spyr(p , 3);
        ly          = (1 - model.spyr(p,1))/scaley + 1;
        deltay      = floor(D(3,i)*scaley);
        sy          = floor(D(3,i)*model.spyr(p,1));
        offsety     = max(0 , ( floor(D(3,i) - ( (ly-1)*deltay + sy + 1)) ));
        
        scalex      = model.spyr(p , 4);
        lx          = (1 - model.spyr(p,2))/scalex + 1;
        deltax      = floor(D(3,i)*scalex);
        sx          = floor(D(3,i)*model.spyr(p,2));
        offsetx     = max(0 , ( floor(D(3,i) - ( (lx-1)*deltax + sx + 1)) ));

		ratio       = maxfactor/(model.spyr(p , 1)*model.spyr(p , 2));

        
        for l = 1 : lx
            origx = offsetx + (l-1)*deltax + 0;
            for m = 1 : ly
                origy                       = offsety + (m-1)*deltay + 0;        
                y                           = D(2,i) + cte*1 + origy;
                x                           = D(1,i) + cte*1 + origx;
                h                           = sy;
				w                           = sx;
                h1                          = h-1;
				w1                          = w-1;
                x1                          = x-1;
                y1                          = y-1;
                
                for j = 1 : size(IIR , 3)
                    H1(co , i)              = area(R(: , : , j) , y , x , h , w)*ratio ;
                    H2(co , i)              = (IIR(y+h1,x+w1,j) + IIR(y1,x1,j) - (IIR(y+h1,x1,j)+IIR(y1,x+w1,j)))*ratio;
                    co                      = co + 1;
                end
            end
        end
    end
end

plot(1:size(HI,1) , H(: , 1) , 1:size(HI,1) , H1(: , 1) , 'r' , 1:size(HI,1) , H2(: , 1) , 'k')

[H(: , 1) , H1(: , 1) , H2(: , 1)]


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 02/26/2011

 Reference [1] Eanes Torres Pereira, Herman Martins Gomes, João Marques de Carvalho 
              "Integral Local Binary Patterns: A Novel Approach Suitable for Texture-Based Object Detection Tasks"
              2010 23rd SIBGRAPI Conference on Graphics, Patterns and Images

		   [2] S. Paris and H. Glotin and Z-Q. Zhao,
		       "Real-time face detection using Integral Histogram of Multi-Scale Local Binary Patterns", ICIC 2011

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
	double          *dimsIscan;
	int             ny;
	int             nx;
	int             maptable;
	int             postprocessing;
	double         *scalingbox;
	int             max_detections;
	double         *mergingbox;

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
int eval_hmblgp_spyr_subwindow(unsigned int * , double * , int , int , int , int , int , int , int , int , double , double , struct model , double *);
int eval_hmblgp_spyr_subwindow_hom(unsigned int * , double * , int , int , int , int , int , int , int , int , double , double , struct model , double *);
void homkertable(struct model  , double * );

#ifdef matfx
double * detector_mlhmslgp_spyr(unsigned char * , int  , int  , struct model  ,  int * , double * , double * );
#else
double * detector_mlhmslgp_spyr(unsigned char * , int  , int  , struct model  ,  int * , double * );
#endif

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *I;
	struct model detector;
	const int *dimsI ;
	int numdimsI;
	double *D , *Dtemp=NULL , *stat;
	double scalingbox_default[3]    = {2 , 1.4 , 1.8};
	double mergingbox_default[3]    = {1/2 , 1/2 , 0.8};
	double norm_default[3]          = {0 , 0 , 4};
	mxArray *mxtemp;	    
	int i , Ny , Nx , nD = 0 , tempint , r = 5 , Nbins = 256;
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
	detector.postprocessing = 1;
	detector.ny             = 24;
	detector.nx             = 24;
	detector.max_detections = 500;

#ifdef OMP 
    detector.num_threads    = -1;
#endif

	if ((nrhs < 1) || (nrhs > 2) )       
	{	
		mexPrintf(
			"\n"
			"\n"
			"\n"
			"Object detector based on fast Histogram of MBLGP features through Histogram Integral and trained by Linear SVM\n" 
			"\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"\n"
#ifdef matfx
			"[D , stat , matfx] = detector_mlhmslgp_spyr(I , [model]);\n"
#else
			"[D , stat] = detector_mlhmslgp_spyr(I , [model]);\n"
#endif
			"\n"
			"\n"  
			"Inputs\n"
			"------\n"
			"\n"
			"I                                     Input image (Ny x Nx) in UINT8 format.\n"
			"\n"
			"model                                 Trained model structure.\n"
			"           w                          Trained model with a Linear SVM, weight vector (1 x ((1+improvedLGP)*Nbins*nH*nscale+addbias)).\n"
			"                                      where Nbins = ([256,59,36,10]*(improvedLGP+1)) if cs_opt = 0, Nbins = ([16,15,10,10]*(improvedLGP+1)) if cs_opt = 1.\n"
			"           addbias                    Add bias or not for model prediction (1/0).\n"
			"           homtable                   Precomputed table for homogeneous additive Kernel approximation (used when model.n > 0).\n"
			"           n                          Order approximation for the homogeneous additive Kernel.\n"
			"           L                          Sampling step (default L = 0.5).\n"
			"           kerneltype                 0 for intersection kernel, 1 for Jensen-shannon kernel, 2 for Chi2 kernel (default kerneltype = 0).\n"
			"           numsubdiv                  Number of subdivisions (default numsubdiv = 8).\n"
			"           minexponent                Minimum exponent value (default minexponent = -20).\n"
			"           maxexponent                Maximum exponent value (default minexponent = 8).\n"
			"           spyr                       Spatial Pyramid (nspyr x 4) (default [1 , 1 , 1 , 1] with nspyr = 1)\n"
			"                                      where spyr(i,1) is the ratio of ny in y axis of the blocks at level i (by(i) = spyr(i,1)*ny)\n"
			"                                      where spyr(i,2) is the ratio of nx in x axis of the blocks at level i (bx(i) = spyr(i,3)*nx)\n"
			"                                      where spyr(i,3) is the ratio of ny in y axis of the shifting at level i (deltay(i) = spyr(i,2)*ny)\n"
			"                                      where spyr(i,4) is the ratio of nx in x axis of the shifting at level i (deltax(i) = spyr(i,4)*nx)\n"
			"                                      where spyr(i,5) is the weight's histogram associated to current level pyramid (w(i) = spyr(i,1)*spyr(i,2))\n"
			"                                      total number of subwindows nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1)).\n"
			"           nH                         Number of subwindows associated with spyr (default nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1))).\n"
			"           scale                      Multi-Scale vector (1 x nscale) (default scale = 1) where scale(i) = s is the size's factor to apply to each 9 blocks\n"
			"                                      in the LGP computation, i = 1,...,nscale.\n"
			"           cs_opt                     Center-Symetric LGP : 1 for computing CS-MBLGP features, 0 : for MBLGP (default cs_opt = 0).\n"
			"           improvedLGP                0 for default 8 bits encoding, 1 for 9 bits encoding (8 + central area).\n"
			"           norm                       Normalization vector (1 x 3) : [for all subwindows, for each subwindows of a pyramid level, for each subwindows]\n"
			"                                      norm = 0 <=> no normalization, norm = 1 <=> v=v/(sum(v)+epsi), norm = 2 <=> v=v/sqrt(sum(v²)+epsi²),\n"
			"                                      norm = 3 <=> v=sqrt(v/(sum(v)+epsi)) , norm = 4 <=> L2-clamped (default norm = [0 , 0 , 4]).\n"
			"           clamp                      Clamping value (default clamp = 0.2).\n"
			"           maptable                   Mapping table for LGP codes. LGP code belongs to {0,...,b}, where b is defined according to following table:\n"
			"                                      |maptable | cs_opt = 0, improvedLGP = 0 | cs_opt = 0, improvedLGP = 1 | cs_opt = 1, improvedLGP = 0 | cs_opt = 1, improvedLGP = 1|\n"
			"                                      |   0     |           255               |              511            |            15               |              31            |\n"
			"                                      |   1     |           58                |              117            |            14               |              29            |\n"
			"                                      |   2     |           35                |              71             |            5                |              11            |\n"
			"                                      |   3     |           9                 |              19             |            5                |              11            |\n"
			"           rmextremebins              Force to zero bin = {0 , b} if  rmextremebins = 1 where b is defined in previous tab (default rmextremebins = 1).\n"
			"           postprocessing             Type of postprocessing in order to reduce false alarms (default postprocessing = 1):\n"
			"                                      0: no postprocessing, i.e. raw detections, 1: merging if rectangles overlapp more than 25%\n"
			"                                      2 : Better Merging detections algorithm with parameters defined by mergingbox.\n"
			"           dimsIscan                  Initial Size of the scanning windows, i.e. (ny x nx ) (default dimsIscan = [24 , 24]).\n"
			"           scalingbox                 [scale_ini , scale_inc , step_ini] where :\n"
			"                                      scale_ini is starting scale factor for each subwindows to apply from the size of trained images (default scale_ini = 2),\n"
			"                                      scale_inc is Increment of the scale factor (default scale_inc = 1.4),\n"
			"                                      step_ini  is the overlapping subwindows factor such that delta = Round(step_ini*scale_ini*scale_inc^(s)) where s in the number of scaling steps (default step_ini = 2).\n"
			"           mergingbox                 [overlap_same , overlap_diff , step_ini]\n"
			"                                      overlap_same is the overlapping factor for merging detections of the same size (first step) (default overlap_same = 1/2),\n"
			"                                      overlap_diff is the overlapping factor for merging detections of the different size (second step) (default overlap_diff = 1/2),\n"
			"                                      dist_ini is the size fraction of the current windows allowed to merge included subwindows (default dist_ini = 1/3).\n"
			"           max_detections             Maximum number of raw subwindows detections (default max_detections = 500).\n"
#ifdef OMP 
			"           num_threads                Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1).\n"
#endif
			"\n"
			"\n"
			"Outputs\n"
			"-------\n"
			"\n"
			"D                                     Detection result (5 x nD) where nD is the number of detection\n"
			"                                      D(1,:) x coordinates of the detections,\n"
			"                                      D(2,:) y coordinates of the detections,\n"
			"                                      D(3,:) size of detection windows,\n"
			"                                      D(4,:) number of merged detection,\n"
			"                                      D(5,:) detection'values.\n"
			"\n"
			"stat                                  Number of positives and negatives detection of all scanned subwindows(1 x 2).\n"
#ifdef matfx
			"\n"
			"matfx                                 Matrix of raw detections (Ny x Nx).\n"
#endif
			);
		return;
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
			detector.nH                   = number_histo_lbp(detector.spyr , detector.nspyr , detector.nscale);
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "dimsIscan" );
		if(mxtemp != NULL)
		{
			detector.dimsIscan            =  mxGetPr(mxtemp);              			
			detector.ny                   = (int)detector.dimsIscan[0];
			detector.nx                   = (int)detector.dimsIscan[1];

			if ((Ny < detector.ny ) || (Nx < detector.nx ))       
			{
				mexErrMsgTxt("I must be at least nyxnx");	
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "cs_opt");
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

			if(detector.cs_opt)
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
			if(detector.cs_opt)
			{
				detector.improvedLGP      = 0;
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

		mxtemp                            = mxGetField( prhs[1] , 0, "postprocessing" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 2))
			{						
				detector.postprocessing   = 1;	
			}
			else
			{
				detector.postprocessing   = tempint;	
			}			
		}	

		mxtemp                            = mxGetField( prhs[1] , 0, "scalingbox" );
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != 3)
			{	
				mexErrMsgTxt("scalingbox must be (1 x 3)");
			}
			detector.scalingbox           = mxGetPr(mxtemp);
		}
		else
		{
			detector.scalingbox           = (double *)mxMalloc(3*sizeof(double));
			for(i = 0 ; i < 3 ; i++)
			{
				detector.scalingbox[i]    = scalingbox_default[i];
			}
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "mergingbox" );
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != 3)
			{	
				mexErrMsgTxt("mergingbox must be (1 x 3)");
			}
			detector.mergingbox           = mxGetPr(mxtemp);
		}
		else
		{
			detector.mergingbox           = (double *)mxMalloc(3*sizeof(double));
			for(i = 0 ; i < 3 ; i++)
			{
				detector.mergingbox[i]    = mergingbox_default[i];
			}
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "max_detections" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];			
			if((tempint < 0))
			{								
				detector.max_detections   = 5000;
			}
			else
			{
				detector.max_detections   = tempint;	
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

		detector.scalingbox            = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.scalingbox[i]     = scalingbox_default[i];
		}

		detector.mergingbox            = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.mergingbox[i]     = mergingbox_default[i];
		}

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

	plhs[1]                            = mxCreateDoubleMatrix(1 , 2 , mxREAL);
	stat                               = mxGetPr(plhs[1]);

	/*------------------------ Main Call ----------------------------*/

#ifdef matfx
	plhs[2]                            = mxCreateDoubleMatrix(Ny , Nx , mxREAL);
	fxmat                              = mxGetPr(plhs[2]);
	Dtemp                              = detector_mlhmslgp_spyr(I , Ny , Nx  , detector  , &nD , stat , fxmat);
#else
	Dtemp                              = detector_mlhmslgp_spyr(I , Ny , Nx  , detector  , &nD , stat);
#endif

	/*----------------------- Outputs -------------------------------*/

	plhs[0]                            = mxCreateDoubleMatrix(r , nD , mxREAL);
	D                                  = mxGetPr(plhs[0]);

	for(i = 0 ; i < r*nD ; i++)
	{	
		D[i]                           = Dtemp[i];	
	}

	/*--------------------------- Free memory -----------------------*/

	free(Dtemp);

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
		if ( mxGetField( prhs[1] , 0 , "scalingbox" ) == NULL )	
		{
			mxFree(detector.scalingbox);
		}
		if ( mxGetField( prhs[1] , 0 , "mergingbox" ) == NULL )	
		{
			mxFree(detector.mergingbox);
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
		mxFree(detector.scalingbox);
		mxFree(detector.mergingbox);
		mxFree(detector.w);
        mxFree(detector.norm);
		if(detector.n > 0)
		{
			mxFree(detector.homtable);
		}
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
#ifdef matfx
double * detector_mlhmslgp_spyr(unsigned char *I , int Ny , int Nx  , struct model detector , int *nD , double *stat , double *fxmat)
#else
double * detector_mlhmslgp_spyr(unsigned char *I , int Ny , int Nx  , struct model detector , int *nD , double *stat )
#endif
{
	double *scalingbox = detector.scalingbox , *mergingbox = detector.mergingbox; 
	double *D , *Draw , *H;
	unsigned int *IIR , *II , *Itemp;
	unsigned char *R;
	double *possize;
	int *indexsize;

	double scale_ini    = scalingbox[0] , scale_inc = scalingbox[1] , step_ini = scalingbox[2];
	double overlap_same = mergingbox[0] , overlap_diff = mergingbox[1] , dist_ini = mergingbox[2];
	double si , sj , sij;
	int nscale = detector.nscale , nH = detector.nH;
	/* int nspyr = detector.nspyr; */
	int ny = detector.ny , nx = detector.nx , NyNx = Ny*Nx , postprocessing = detector.postprocessing , n = detector.n;
	int maptable = detector.maptable , cs_opt = detector.cs_opt , improvedLGP = detector.improvedLGP;
	int sizeDataBase = max(nx , ny), halfsizeDataBase = sizeDataBase/2 , current_sizewindow , current_stepwindow;
	int Pos_current = detector.max_detections, Pos=0 , Pos1, Negs=0 , ind = 0 , index = 0 , indi , indj,minN = min(Ny,Nx);
#ifdef OMP 
	int num_threads = detector.num_threads;
#endif
	int i , j , l , m , v ;
	int yest , Deltay , Deltax , Ly , Lx , Offsety , Offsetx , Origy , Origx , nys, nxs , r = 5;
	int Nbins , Nbinsnscale , NbinsnscalenH , powN = 256;

	double tempx , tempy, scale_win , powScaleInc , dsizeDataBase = (double) sizeDataBase;
	double tmp , nb_detect_total , nb_detect , nb_detect1, Xinf, Yinf, Xsup, Ysup , fx , maxfactor = 0.0;

	unsigned int table_normal_8[256] = {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 , 55 , 56 , 57 , 58 , 59 , 60 , 61 , 62 , 63 , 64 , 65 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 , 96 , 97 , 98 , 99 , 100 , 101 , 102 , 103 , 104 , 105 , 106 , 107 , 108 , 109 , 110 , 111 , 112 , 113 , 114 , 115 , 116 , 117 , 118 , 119 , 120 , 121 , 122 , 123 , 124 , 125 , 126 , 127 , 128 , 129 , 130 , 131 , 132 , 133 , 134 , 135 , 136 , 137 , 138 , 139 , 140 , 141 , 142 , 143 , 144 , 145 , 146 , 147 , 148 , 149 , 150 , 151 , 152 , 153 , 154 , 155 , 156 , 157 , 158 , 159 , 160 , 161 , 162 , 163 , 164 , 165 , 166 , 167 , 168 , 169 , 170 , 171 , 172 , 173 , 174 , 175 , 176 , 177 , 178 , 179 , 180 , 181 , 182 , 183 , 184 , 185 , 186 , 187 , 188 , 189 , 190 , 191 , 192 , 193 , 194 , 195 , 196 , 197 , 198 , 199 , 200 , 201 , 202 , 203 , 204 , 205 , 206 , 207 , 208 , 209 , 210 , 211 , 212 , 213 , 214 , 215 , 216 , 217 , 218 , 219 , 220 , 221 , 222 , 223 , 224 , 225 , 226 , 227 , 228 , 229 , 230 , 231 , 232 , 233 , 234 , 235 , 236 , 237 , 238 , 239 , 240 , 241 , 242 , 243 , 244 , 245 , 246 , 247 , 248 , 249 , 250 , 251 , 252 , 253 , 254 , 255};
	unsigned int table_u2_8[256]     = {0 , 1 , 2 , 3 , 4 , 58 , 5 , 6 , 7 , 58 , 58 , 58 , 8 , 58 , 9 , 10 , 11 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 12 , 58 , 58 , 58 , 13 , 58 , 14 , 15 , 16 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 17 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 18 , 58 , 58 , 58 , 19 , 58 , 20 , 21 , 22 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 23 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 24 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 25 , 58 , 58 , 58 , 26 , 58 , 27 , 28 , 29 , 30 , 58 , 31 , 58 , 58 , 58 , 32 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 33 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 34 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 35 , 36 , 37 , 58 , 38 , 58 , 58 , 58 , 39 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 40 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 41 , 42 , 43 , 58 , 44 , 58 , 58 , 58 , 45 , 58 , 58 , 58 , 58 , 58 , 58 , 58 , 46 , 47 , 48 , 58 , 49 , 58 , 58 , 58 , 50 , 51 , 52 , 58 , 53 , 54 , 55 , 56 , 57};
	unsigned int table_ri_8[256]     = {0 , 1 , 1 , 2 , 1 , 3 , 2 , 4 , 1 , 5 , 3 , 6 , 2 , 7 , 4 , 8 , 1 , 9 , 5 , 10 , 3 , 11 , 6 , 12 , 2 , 13 , 7 , 14 , 4 , 15,8,16,1,5,9,13,5,17,10,18,3,17,11,19,6,20,12,21,2,10,13,22,7,23,14,24,4,18,15,25,8,26,16,27,1,3,5,7,9,11,13,15,5,17,17,20,10,23,18,26,3,11,17,23,11,28,19,29,6,19,20,30,12,29,21,31,2,6,10,14,13,19,22,25,7,20,23,30,14,30,24,32,4,12,18,24,15,29,25,33,8,21,26,32,16,31,27,34,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,5,13,17,18,17,19,20,21,10,22,23,24,18,25,26,27,3,7,11,15,17,20,23,26,11,23,28,29,19,30,29,31,6,14,19,25,20,30,30,32,12,24,29,33,21,32,31,34,2,4,6,8,10,12,14,16,13,18,19,21,22,24,25,27,7,15,20,26,23,29,30,31,14,25,30,32,24,33,32,34,4,8,12,16,18,21,24,27,15,26,29,31,25,32,33,34,8,16,21,27,26,31,32,34,16,27,31,34,27,34,34,35};	
	unsigned int table_riu2_8[256]   = {0 , 1 , 1 , 2 , 1 , 9 , 2 , 3 , 1 , 9 , 9 , 9 , 2 , 9 , 3 , 4 , 1 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 2 , 9 , 9 , 9 , 3 , 9 , 4 , 5 , 1 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 2 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 3 , 9 , 9 , 9 , 4 , 9 , 5 , 6 , 1 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 2 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 3 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 4 , 9 , 9 , 9 , 5 , 9 , 6 , 7 , 1 , 2 , 9 , 3 , 9 , 9 , 9 , 4 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 5 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 6 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 7 , 2 , 3 , 9 , 4 , 9 , 9 , 9 , 5 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 6 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 7 , 3 , 4 , 9 , 5 , 9 , 9 , 9 , 6 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 7 , 4 , 5 , 9 , 6 , 9 , 9 , 9 , 7 , 5 , 6 , 9 , 7 , 6 , 7 , 7 , 8};

	unsigned int table_normal_4[16]  = {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 };
	unsigned int table_u2_4[16]      = {0 , 1 , 2 , 3 , 4 , 14 , 5 , 6 , 7 , 8 , 14 , 9 , 10 , 11 , 12 , 13};
	unsigned int table_ri_4[16]      = {0 , 1 , 1 , 2 , 1 , 3 , 2 , 4 , 1 , 2 , 3 , 4 , 2 , 4 , 4 , 5};	
	unsigned int table_riu2_4[16]    = {0 , 1 , 1 , 2 , 1 , 5 , 2 , 3 , 1 , 2 , 5 , 3 , 2 , 3 , 3 , 4};
	unsigned int *table;

#ifdef matfx
	int indOrigx;
#endif

	if(cs_opt == 1)
	{
		powN                            = 16;
		if(maptable == 0)
		{
			Nbins                       = 16;
		}
		else if(maptable == 1)
		{
			Nbins                       = 15;
		}
		else if(maptable == 2)
		{
			Nbins                       = 6;
		}
		else if(maptable == 3)
		{
			Nbins                       = 6;
		}
	}
	else
	{
		if(maptable == 0)
		{
			Nbins                       = 256;
		}
		else if(maptable == 1)
		{
			Nbins                       = 59;
		}
		else if(maptable == 2)
		{
			Nbins                       = 36;
		}
		else if(maptable == 3)
		{
			Nbins                       = 10;
		}
	}

	if(improvedLGP == 1)
	{
		Nbins                      *= 2;
	}

	Nbinsnscale                     = Nbins*nscale;
	NbinsnscalenH                   = Nbinsnscale*nH;

	IIR                             = (unsigned int *) malloc(NyNx*Nbinsnscale*sizeof(unsigned int));
	R                               = (unsigned char *) malloc(NyNx*Nbinsnscale*sizeof(unsigned char));
	II                              = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Draw                            = (double *) malloc(r*Pos_current*sizeof(double));
	table                           = (unsigned int *) malloc((powN*(improvedLGP+1))*sizeof(unsigned int));

#ifdef OMP 

#else
	Itemp                           = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	H                               = (double *) malloc(NbinsnscalenH*sizeof(double));
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
	Itemp                          = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
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
		Itemp                      = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
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

	current_sizewindow              = halfsizeDataBase*Round(2.0*scale_ini);	
	current_stepwindow              = Round(step_ini*scale_ini);
	powScaleInc                     = scale_inc;

	while(current_sizewindow <= minN)  
	{
		scale_win                   = (double) (current_sizewindow) / dsizeDataBase ;

		nys                         = Round(ny*scale_win) + 1;  
		nxs                         = Round(nx*scale_win) + 1;  

		Deltay                      = current_stepwindow ;
		Deltax                      = current_stepwindow ;

		Ly                          = max(1 , (int) (floor(((Ny - nys)/(double) Deltay))) + 1);
		Offsety                     = max(0 , (int)( floor(Ny - ( (Ly-1)*Deltay + nys + 1)) ));

		Lx                          = max(1 , (int) (floor(((Nx - nxs)/(double) Deltax))) + 1);
		Offsetx                     = max(0 , (int)( floor(Nx - ( (Lx-1)*Deltax + nxs + 1)) ));

#ifdef OMP 
#ifdef matfx
#pragma omp parallel default(none) private(m,Origy,yest,fx,index,l,Origx,indOrigx,H) shared(fxmat,Pos,Negs,Pos_current,Lx,Ly,Offsetx,Offsety,Deltax,Deltay,Draw,IIR,Nx,Ny,r,n,scale_win,current_sizewindow,detector,Nbins,NbinsnscalenH,maxfactor) 
#else
#pragma omp parallel default(none) private(m,Origy,yest,fx,index,l,Origx,H) shared(Pos,Negs,Pos_current,Lx,Ly,Offsetx,Offsety,Deltax,Deltay,Draw,IIR,Nx,Ny,r,n,scale_win,current_sizewindow,detector,Nbins,NbinsnscalenH,maxfactor) 
#endif
#endif
		{
#ifdef OMP 
			H                       = (double *) malloc(NbinsnscalenH*sizeof(double));
#else
#endif

#ifdef OMP 
#pragma omp for nowait
#endif
			for(l = 0 ; l < Lx ; l++) 
			{
				Origx          = Offsetx + l*Deltax ;

#ifdef matfx
				indOrigx       = Ny*Origx;
#endif

				for(m = 0 ; m < Ly ; m++)  
				{
					Origy                     = Offsety + m*Deltay ;
					if(n > 0)
					{
						yest                  = eval_hmblgp_spyr_subwindow_hom(IIR , H  , Origy , Origx , Ny , Nx , current_sizewindow , current_sizewindow , Nbins , NbinsnscalenH , scale_win , maxfactor , detector , &fx);
					}
					else
					{
						yest                  = eval_hmblgp_spyr_subwindow(IIR , H  , Origy , Origx   , Ny , Nx , current_sizewindow , current_sizewindow  , Nbins , NbinsnscalenH , scale_win , maxfactor , detector , &fx);
					}

#ifdef matfx
					fxmat[Origy + indOrigx]  += fx;
#endif
					if(yest == 1) 
					{
						if(Pos < Pos_current)
						{
							index                     = Pos*r;
							Draw[0 + index]           = 1.0;  
							Draw[1 + index]           = (double)Origx;				
							Draw[2 + index]           = (double)Origy;
							Draw[3 + index]           = (double)current_sizewindow;
							Draw[4 + index]           = fx;
							Pos++;		
						}
					}	
					else
					{			
						Negs++;	
					}	
				}			
			}

#ifdef OMP
			free(H);
#else

#endif
		}

		current_sizewindow        = halfsizeDataBase*Round(2.0*scale_ini*powScaleInc);	
		current_stepwindow        = (int)ceil(step_ini*scale_ini*powScaleInc);
		powScaleInc              *= scale_inc; 
	}
	if(postprocessing == 0) 
	{
		nD[0]    = Pos;
		D        = (double *) malloc(Pos*r*sizeof(double));
		indi     = 0;

		for(i = 0 ; i < Pos ; i++)
		{
			D[0 + indi] = (Draw[1 + indi] + 1.0);   
			D[1 + indi] = (Draw[2 + indi] + 1.0);  
			D[2 + indi] = Draw[3 + indi]; 
			D[3 + indi] = Draw[0 + indi];
			D[4 + indi] = Draw[4 + indi];
			indi       += r;
		}
	}
	else if(postprocessing == 1)  
	{	
		indi = 0;
		Pos1 = Pos - 1;

		for(i = 0 ; i < Pos1 ; i++)
		{
			current_sizewindow  = (int)Draw[3 + indi];
			current_stepwindow  = Round(current_sizewindow*overlap_same);

			tmp                 = (int)Draw[1 + indi];
			Xinf                = tmp - current_stepwindow;
			Xsup                = tmp + current_stepwindow;

			tmp                 = (int)Draw[2 + indi];
			Yinf                = tmp - current_stepwindow;
			Ysup                = tmp + current_stepwindow;

			indj                = indi + r;

			for(j = i+1 ; j < Pos ;  j++)
			{
				if( (current_sizewindow==(int)Draw[3 + indj]) && (Xinf < Draw[1 + indj]) && (Xsup >= Draw[1 + indj]) && (Yinf < Draw[2 + indj]) && (Ysup >= Draw[2 + indj]))
				{
					nb_detect                 = Draw[0 + indi];
					nb_detect1                = nb_detect + 1.0;
					Draw[1 + indj]            = Round((nb_detect*Draw[1 + indi] + Draw[1 + indj])/nb_detect1);
					Draw[2 + indj]            = Round((nb_detect*Draw[2 + indi] + Draw[2 + indj])/nb_detect1);
					Draw[0 + indj]            = nb_detect1;
					Draw[0 + indi]            = 0.0;
					break;
				}
				indj   += r;
			}
			indi  += r;
		}

		indi = 0;
		for(i = 0 ; i < Pos1 ; i++)
		{
			if(Draw[0 + indi])
			{
				tmp      = Draw[3 + indi]*overlap_diff; 
				Xsup     = Draw[1 + indi] + tmp;
				Ysup     = Draw[2 + indi] + tmp;
				step_ini = Draw[3 + indi]*dist_ini;

				indj     = indi + r;

				for(j = i+1 ; j < Pos ; j++)
				{
					if(Draw[0 + indj])
					{
						tmp                 = Draw[3 + indj]*overlap_diff; 
						tempx               = Xsup - (Draw[1 + indj] + current_sizewindow);
						tempy               = Ysup - (Draw[2 + indj] + current_sizewindow);

						if(sqrt(tempx*tempx + tempy*tempy) <= step_ini)			
						{
							nb_detect       = Draw[0 + indi];
							nb_detect1      = Draw[0 + indj];
							nb_detect_total = nb_detect + nb_detect1;

							Draw[1 + indj]  = Round((nb_detect*Draw[1 + indi] + nb_detect1*Draw[1 + indj])/nb_detect_total);
							Draw[2 + indj]  = Round((nb_detect*Draw[2 + indi] + nb_detect1*Draw[2 + indj])/nb_detect_total);
							Draw[3 + indj]  = Round((nb_detect*Draw[3 + indi] + nb_detect1*Draw[3 + indj])/nb_detect_total);
							Draw[4 + indj]  = (nb_detect*Draw[4 + indi] + nb_detect1*Draw[4 + indj])/nb_detect_total;

							Draw[0 + indj]  = nb_detect_total;
							Draw[0 + indi]  = 0.0;

							break;
						}
					}
					indj  += r;
				}
			}
			indi   += r;
		}

		ind      = 0;
		indi     = 0;
		for(i = 0 ; i < Pos ; i++)
		{
			if(Draw[indi]!=0)
			{
				ind++;
			}
			indi  += r;
		}

		nD[0]    = ind;
		D        = (double *) malloc(ind*r*sizeof(double));

		indi     = 0;
		indj     = 0;
		for(i = 0 ; i < Pos ; i++)
		{
			if(Draw[indi]!=0)
			{
				D[0 + indj] = (Draw[1 + indi] + 1.0);  
				D[1 + indj] = (Draw[2 + indi] + 1.0);  
				D[2 + indj] = Draw[3 + indi]; 
				D[3 + indj] = Draw[0 + indi];
				D[4 + indj] = Draw[4 + indi];				
				indj       += r;
			}
			indi  += r;
		}	
	}
	else if(postprocessing == 2)  
	{	
		indi = 0;
		Pos1 = Pos - 1;
		for(i = 0 ; i < Pos1 ; i++)
		{
			current_sizewindow  = (int)Draw[3 + indi];
			current_stepwindow  = Round(current_sizewindow*overlap_same);

			tmp                 = (int)Draw[1 + indi];
			Xinf                = tmp - current_stepwindow;
			Xsup                = tmp + current_stepwindow;

			tmp                 = (int)Draw[2 + indi];
			Yinf                = tmp - current_stepwindow;
			Ysup                = tmp + current_stepwindow;

			indj                = indi + r;
			for(j = i+1 ; j < Pos ;  j++)
			{
				if( (current_sizewindow==(int)Draw[3 + indj]) && (Xinf < Draw[1 + indj]) && (Xsup >= Draw[1 + indj]) && (Yinf < Draw[2 + indj]) && (Ysup >= Draw[2 + indj]))
				{
					nb_detect                 = Draw[0 + indi];
					nb_detect1                = nb_detect + 1.0;
					Draw[1 + indj]            = Round((nb_detect*Draw[1 + indi] + Draw[1 + indj])/nb_detect1);
					Draw[2 + indj]            = Round((nb_detect*Draw[2 + indi] + Draw[2 + indj])/nb_detect1);
					Draw[0 + indj]            = nb_detect1;
					Draw[0 + indi]            = 0.0;
					break;
				}
				indj   += r;
			}
			indi  += r;
		}

		possize                 = (double *) malloc(Pos*sizeof(double));
		indexsize               = (int *) malloc(Pos*sizeof(int));

		for( i = 0 ; i < Pos ; i++)
		{
			possize[i]         = Draw[3 + i*r];
			indexsize[i]       = i;
		}

		qsindex(possize , indexsize , 0 , Pos1);

		for(i = 0 ; i < Pos1 ; i++)
		{
			indi = indexsize[Pos1-i]*r;
			if(Draw[0 + indi])
			{
				tmp      = Draw[3 + indi]*overlap_diff; 
				Xsup     = Draw[1 + indi] + tmp;
				Ysup     = Draw[2 + indi] + tmp;
				step_ini = Draw[3 + indi]*dist_ini;

				for(j = i+1 ; j < Pos ; j++)
				{
					indj = indexsize[Pos1-j]*r;
					if(Draw[0 + indj])
					{
						tmp                 = Draw[3 + indj]*overlap_diff; 
						tempx               = Xsup - (Draw[1 + indj] + tmp);
						tempy               = Ysup - (Draw[2 + indj] + tmp);

						if(sqrt(tempx*tempx + tempy*tempy) <= step_ini)			
						{
							nb_detect       = Draw[0 + indi];
							si              = Draw[3 + indi];
							si             *= si;
							sj              = Draw[3 + indj];
							sj             *= sj;
							sij             = 1.0/(si + sj);

							Draw[1 + indj]  = Round((si*Draw[1 + indi] + sj*Draw[1 + indj])*sij);
							Draw[2 + indj]  = Round((si*Draw[2 + indi] + sj*Draw[2 + indj])*sij);
							Draw[3 + indj]  = Round((si*Draw[3 + indi] + sj*Draw[3 + indj])*sij);
							Draw[4 + indj]  = (si*Draw[4 + indi] + sj*Draw[4 + indj])*sij;

							Draw[0 + indj]  = nb_detect + 1;
							Draw[0 + indi]  = 0.0;
							break;
						}
					}
				}
			}
		}

		ind      = 0;
		indi     = 0;
		for(i = 0 ; i < Pos ; i++)
		{
			if(Draw[indi]!=0)
			{
				ind++;
			}
			indi  += r;
		}

		nD[0]    = ind;
		D        = (double *) malloc(ind*r*sizeof(double));

		indi     = 0;
		indj     = 0;
		for(i = 0 ; i < Pos ; i++)
		{
			if(Draw[indi]!=0)
			{
				D[0 + indj] = (Draw[1 + indi] + 1.0);  
				D[1 + indj] = (Draw[2 + indi] + 1.0); 
				D[2 + indj] = Draw[3 + indi]; 
				D[3 + indj] = Draw[0 + indi];
				D[4 + indj] = Draw[4 + indi];				
				indj       += r;
			}
			indi  += r;
		}
		free(possize);
		free(indexsize);
	}

	free(IIR);
	free(R);
	free(II);
	free(Draw);
	free(table);

#ifdef OMP
#else
	free(H);
	free(Itemp);
#endif	

	stat[0] = (double)Pos;
	stat[1] = (double)Negs;
	return D;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int eval_hmblgp_spyr_subwindow(unsigned int *IIR , double *H  , int Origy , int Origx , int Ny , int Nx , int nys , int nxs , int Nbins , int  NbinsnscalenH , double scale_win , double maxfactor , struct model detector  , double *fx)
{
	double *w = detector.w , *spyr = detector.spyr;
	double clamp = detector.clamp;
	double scaley, scalex,ratio , sum , temp;
	int nspyr = detector.nspyr , nscale = detector.nscale , rmextremebins = detector.rmextremebins , cs_opt = detector.cs_opt , improvedLGP = detector.improvedLGP;
	int p , l , m , s , i , j;
	int origy, origx, deltay, deltax, sy , sx , ly, lx , offsety = 0 , offsetx = 0 , coNbins = 0;
	int NyNx = Ny*Nx , NyNxNbins , sNyNxNbins , NBINS;
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
		deltay      = (int) (nys*scaley);
		sy          = (int) (nys*spyr[p + 0]);
		offsety     = max(0 , (int) ( floor(nys - ( (ly-1)*deltay + sy + 1)) ));

		scalex      = (spyr[p + nspyr*3]);
		lx          = (int) ( (1 - spyr[p + nspyr*1])/scalex + 1);
		deltax      = (int) (nxs*scalex);
		sx          = (int) (nxs*spyr[p + nspyr*1]);
		offsetx     = max(0 , (int) ( floor(nxs - ( (lx-1)*deltax + sx + 1)) ));

		ratio       = 1.0/spyr[p + nspyr*4];
		co_p        = 0;
		offset      = co_totalp*Nbinsnscale;

		for(l = 0 ; l < lx ; l++) /* Loop shift on x-axis */
		{
			origx  = offsetx + l*deltax + Origx ;
			for(m = 0 ; m < ly ; m++)   /* Loop shift on y-axis  */
			{
				origy     = offsety + m*deltay + Origy;
				for (s = 0 ; s < nscale ; s++)
				{
					sNyNxNbins         = s*NyNxNbins;
					for (i = 0 ; i < Nbins ; i++)
					{
						H[i + coNbins] = Area(IIR + i*NyNx + sNyNxNbins , origx  , origy  , sx , sy , Ny);
					}

					for(i = coNbins ; i < coNbins+Nbins ; i++)
					{
						H[i]          *= ratio;
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
						sum        = 0.0;
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
int eval_hmblgp_spyr_subwindow_hom(unsigned int *IIR , double *H , int Origy , int Origx , int Ny , int Nx , int nys , int nxs , int Nbins , int  NbinsnscalenH , double scale_win , double maxfactor , struct model detector  , double *fx)
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
	int NyNx = Ny*Nx , NyNxNbins , sNyNxNbins , NBINS;
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
		deltay      = (int) (nys*scaley);
		sy          = (int) (nys*spyr[p + 0]);
		offsety     = max(0 , (int) ( floor(nys - ( (ly-1)*deltay + sy + 1)) ));

		scalex      = (spyr[p + nspyr*3]);
		lx          = (int) ( (1 - spyr[p + nspyr*1])/scalex + 1);
		deltax      = (int) (nxs*scalex);
		sx          = (int) (nxs*spyr[p + nspyr*1]);
		offsetx     = max(0 , (int) ( floor(nxs - ( (lx-1)*deltax + sx + 1)) ));

		ratio       = 1.0/spyr[p + nspyr*4];
		co_p        = 0;
		offset      = co_totalp*Nbinsnscale;

		for(l = 0 ; l < lx ; l++) /* Loop shift on x-axis */
		{
			origx  = offsetx + l*deltax + Origx ;
			for(m = 0 ; m < ly ; m++)   /* Loop shift on y-axis  */
			{
				origy     = offsety + m*deltay + Origy;
				for (s = 0 ; s < nscale ; s++)
				{
					sNyNxNbins         = s*NyNxNbins;

					for (i = 0 ; i < Nbins ; i++)
					{
						H[i + coNbins] = Area(IIR + i*NyNx + sNyNxNbins , origx  , origy  , sx , sy , Ny);
					}

					for(i = coNbins ; i < coNbins+Nbins ; i++)
					{
						H[i]          *= ratio;
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
						sum        = 0.0;
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
						if(fabs(sumA -  (8.0*Area(II , xc  , yc  , currentscale , currentscale , Ny)))> 3.0*gn )
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
					if(A5 > gn)
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
		nH         += ly*lx;
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
