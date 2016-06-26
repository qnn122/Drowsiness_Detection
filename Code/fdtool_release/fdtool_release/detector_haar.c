/*

  Object detector by boosting Haar features (Gentleboosting & Adaboosting) 

  Usage
  ------

  [D , stat , [matfx]] = detector_haar(I , [model]);

  
  Inputs
  -------

  I                                     Input image (Ny x Nx) in UINT8 format
  
  model                                 Trained model structure

             weaklearner                Choice of the weak learner used in the training phase (default weaklearner = 2)
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
             param                      Trained parameters matrix (4 x T) in double format. Each row correspond to :
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)
			                            b                         Offset (1 x T)
			 dimsItraining              Size of the train images used in the mblbp computation, i.e. (ny x nx ) (Default dimsItraining = [24 x 24])
             rect_param                 Feature's rectangles dictionnary (10 x nR) in double format
			 F                          Feature's parameters (6 x nF) in UINT32 format
             cascade_type               Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)
             cascade                    Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
										If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))
										cascade(2 , :) reprensent thresholds for each segment
			 max_detections             Maximum number of raw subwindows detections (default max_detections = 500)
             scalingbox                 [scale_ini , scale_inc , step_ini] where :
                                        scale_ini is starting scale factor for each subwindows to apply from the size of trained images (default scale_ini = 2)
                                        scale_inc is Increment of the scale factor (default scale_inc = 1.4)
					                    step_ini  is the overlapping subwindows factor such that delta = Round(step_ini*scale_ini*scale_inc^(s)) where s in the number of scaling steps (default step_ini = 2)
             postprocessing             Type of postprocessing in order to reduce false alarms (default postprocessing = 1): 
			                            0: no postprocessing, i.e. raw detections, 1: merging if rectangles overlapp more than 25%
										2 : Better Merging detections algorithm with parameters defined by mergingbox
			 mergingbox                 [overlap_same , overlap_diff , step_ini]
                                        overlap_same is the overlapping factor for merging detections of the same size (first step) (default overlap_same = 1/2)
                                        overlap_diff is the overlapping factor for merging detections of the different size (second step) (default overlap_diff = 1/2)
					                    dist_ini is the size fraction of the current windows allowed to merge included subwindows (default dist_ini = 1/3)
			 max_detections             Maximum number of raw subwindows detections (default max_detections = 500)
          
If compiled with the "OMP" compilation flag
			
			num_threads                 Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  D                                     Detection results (5 x nD) where nD is the number of detections
                                        D(1,:) x coordinates of the detections
                                        D(2,:) y coordinates of the detections
                                        D(3,:) size of detection windows
										D(4,:) number of merged detection
										D(5,:) detection'values

  stat                                  Number of positives and negatives detection of all scanned subwindows(1 x 2)

  If compiled with the "matfx" compilation flag, ouputs are

  matfx                                 Matrix of raw detections (Ny x Nx)


  To compile
  ----------


  mex  -output detector_haar.dll detector_haar.c

  mex -g  -output detector_haar.dll detector_haar.c

  mex  -f mexopts_intel10.bat -output detector_haar.dll detector_haar.c


  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output detector_haar.dll detector_haar.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

  or with the matfx directive

  mex -Dmatfx -f mexopts_intel10.bat -output detector_haar.dll detector_haar.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -Dmatfx -DOMP -f mexopts_intel10.bat -output detector_haar.dll detector_haar.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


  Example 1
  ---------


  close all
  load('model_detector_haar_24x24.mat');

  model.cascade_type           = 0;
  model.postprocessing         = 1;
%  model.cascade               = [2 , 8 , 10 , 20 , 30 , 30 ; 0 , 0 ,  0 ,  0 , 0 , 0];
  model.cascade                = [3 , 7 , 10 , 10 , 20 , 20 , 30 ; -0.5 , -0.5 , -0.25 ,  0 ,  0 , 0 , 0];
%  model.cascade               = [1 , 2 , 3 , 4 , 10 , 10 , 20 , 20 , 30 ; -1 , -0.5 , -0.5 , -0.5 , -0.25 ,  0 ,  0 , 0 , 0];

  model.scalingbox            = [1.5 , 1.3 , 2];
  model.mergingbox            = [1/2 , 1/2 , 0.8];
  min_detect                  = 2;

%  I                           = (rgb2gray(imread('class57.jpg')));
  I                           = (rgb2gray(imread('2.bmp')));

 
  tic,[D , stat]              = detector_haar(I , model);,toc

  figure, imshow(I);
  hold on;
  h = plot_rectangle(D , 'r');
  hold off
  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))



  figure, imshow(I);
  hold on;
  h = plot_rectangle(D(: , (D(4 , :) >=min_detect)) , 'g');
  hold off
  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))


  Example 2
  ---------

  if compiled with the matfx pragma

  close all
  load('model_detector_haar_24x24.mat');
  model.cascade_type          = 1;
  model.cascade               = [2 , 8 , 10 , 20 , 30 , 30 ; 0 , 0 ,  0 ,  0 , 0 , 0];
  model.scalingbox            = [1.5 , 1.3 , 2];
  min_detect                  = 2;

  I                           = (rgb2gray(imread('class57.jpg')));

  tic,[D , stat , matfx]      = detector_haar(I , model);,toc
  [indy , indx]               = find(matfx); 
 
  rect                        = [D(1 , :)' , D(2 , :)' , D(3 , :)' , D(3 , :)'];
  [z,its]                     = size(D);

  figure, imshow(I);
  hold on;

  for i=1:its
    if(D(4,i)<min_detect)
        rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[1,0,0],'LineWidth',2);
    else
        rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    end
    text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);
  end
  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))
  plot3(indx , indy , find(matfx) , '+')

  hold off

  figure, imshow(I);
  hold on;
  for i=1:its
  if(D(4,i)>=min_detect)
    rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);

  end
  end
  plot3(indx , indy , find(matfx) , '+')
  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))
  hold off



  Example 3
  ---------

  clear,close all

  load('model_detector_haar_24x24.mat');

%  haar                        = load('model_detector_haar_24x24_wl2_ct0_nP19.mat');
%  haar                        = load('model_detector_haar_24x24_wl2_ct1_nP2.mat');
%  haar                        = load('model_detector_haar_24x24_wl2_ct1_nP2_2.mat');
%  min_detect                  = 1;

   model.cascade               = [2 , 8 , 10 , 20 , 30 , 30 ; 0 , 0 ,  0 ,  0 , 0 , 0];
%  cascade                     = [5 4 4 6 7 ;0 0 0 0 0 ];
%  min_detect                  = 1;


  model.scalingbox             = [2 , 1.4 , 2];

  min_detect                  = 2;


  vid = videoinput('winvideo' , 1 , 'RGB24_320x240');
  vid = videoinput('winvideo' , 1 , 'RGB24_1280x960');
  vid = videoinput('winvideo' , 1 , 'RGB24_640x480');

  preview(vid);
  fig1 = figure(1);
  set(fig1,'doublebuffer','on');
  while(1)
    aa   = getsnapshot(vid);
    pos  =  detector_haar(rgb2gray(aa) , haar.model , scalingbox , cascade);

    image(aa);
    hold on
    for i=1:size(pos,2)
	   if(pos(4 , i) >= min_detect)
        rectangle('Position', [pos(1,i),pos(2,i),pos(3,i),pos(3,i)], 'EdgeColor', [0,1,0], 'linewidth', 2);
		end
    end
    hold off

    drawnow;
  end



  Example 4
  ---------

  clear,close all

  aa                                = vcapg2(0,2);

  load('model_detector_haar_24x24.mat');

  model.cascade                     = [2 , 8 , 10 , 20 , 30 , 30 ; 0 , 0 ,  0 ,  0 , 0 , 0];
  model.cascade                     = [3 , 7 , 10 , 20 , 30 , 30 ; -0.5, -0.25,  0 ,  0 , 0 , 0]; %(640x480)
  model.cascade                     = [1 , 2 , 7 , 10 , 20 , 30 , 30 ; -0.75 , -0.5, -0.25,  0 ,  0 , 0 , 0]; %(640x480)
  model.cascade                     = [1 , 2 , 3 , 4 , 10 , 20 , 30 , 30 ; -0.75 ,-0.5 , -0.5, -0.25,  0 ,  0 , 0 , 0];
  model.scalingbox                  = [2 , 1.4 , 1.8];

  min_detect                        = 2;


  fig1 = figure(1);
  set(fig1,'doublebuffer','on');
  while(1)
    t1   = clock;
    aa   = vcapg2(0,0);
    pos  =  detector_haar(rgb2gray(aa) , model);

    image(aa);
    hold on
    h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g');    
	hold off
	t2   = etime(clock,t1);
	title(sprintf('Fps = %6.3f' , 1/t2));

    drawnow;
  end


  Example 6   if compiled with the matfx option
  ---------


  close all
  load('model_detector_haar_24x24.mat');

  model.cascade_type                 = 1;
  model.postprocessing              = 0;

  model.cascade                     = [2 , 8 , 10 , 20 , 30 , 30 ; 0 , 0 ,  0 ,  0 , 0 , 0];
  model.cascade                     = [10 , 10 , 10 , 20 , 20 , 30 ;   -0.5 , -0.25 ,  0 ,  0 , 0 , 0];
  model.scalingbox                  = [1.2 , 1.2 , 1];
  
  proba                       = 0.99;
  I                           = (rgb2gray(imread('class57.jpg')));
 
  tic,[D , stat , fx]         = detector_haar(I , model);,toc

  px                          = 1./(1+exp(-2.*fx));

  [indy , indx]               = find(px > proba);



  imagesc(px)
  hold on
  plot(indx , indy , 'mo' , 'markersize' , 8)
  hold off
  title('Pr(y=1|x)')




  Example 7
  ---------


  clear,close all
  load('model_detector_haar_24x24.mat');

  model.cascade_type          = 1;
  model.postprocessing        = 0;
  model.cascade               = [2 , 8 , 10 , 20 , 30 , 30 ; 0 , 0 ,  0 ,  0 , 0 , 0];
  %cascade                     = [3 , 7 , 10 , 10 , 20 , 20 , 30 ; -0.5 , -0.5 , -0.25 ,  0 ,  0 , 0 , 0];
  model.scalingbox            = [1.5 , 1.3 , 2];
  
  min_detect                  = 2;
  nb_wobble                   = 32;

  I                           = (rgb2gray(imread('class57.jpg')));
  [Ny , Nx]                   = size(I);
  [x , y]                     = ndgrid((1:Nx) , (1:Ny));
  z                           = [x(:) , y(:)]';

  D                           = [];

  for i = 1 : nb_wobble

  W                           = eye(2) + 0.025*(rand(2)-0.5);
  invW                        = inv(W);
  zi                          = W*z;
  xi                          = reshape(zi(1 , :) , Nx , Ny);
  yi                          = reshape(zi(2 , :) , Nx , Ny);
  Iw                          = interp2(x' , y' , double(I) , xi' , yi');
  Iw(isnan(Iw))               = 0;
  Iw                          = uint8(Iw);

 
  tic,[pos , stat]            = detector_haar(Iw ,  model );,toc
  %pos(1:2 , :)                = invW*pos(1:2 , :);
  D                           = [D , pos];

  end

  figure, imshow(I);
  hold on;
  h = plot(D(1 , :) , D(2 , :) , 'r+');
  hold off
  %title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))



  figure, imshow(I);
  hold on;
  h = plot_rectangle(D(: , find(D(4 , :) >=min_detect)) , 'g');
  hold off
  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))




 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 02/20/2009

 Reference ""


*/

#include <math.h>
#include <mex.h>


#ifdef OMP 
 #include <omp.h>
#endif

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

#define sign(a)    ((a) >= (0) ? (1.0) : (-1.0))
 

struct model
{
	int            weaklearner;
	double         epsi;
	double        *param;
	int            T;
	double        *dimsItraining;
	int            ny;
	int            nx;
	double        *rect_param;
	int            nR;
	unsigned int  *F;
	int            nF;
	int            postprocessing;
	double        *scalingbox;
	int            cascade_type;
	double        *cascade;
	int            Ncascade;
	int            max_detections;
	double        *mergingbox;

#ifdef OMP 
    int            num_threads;
#endif
};

/*------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

int Round(double);
int number_haar_features(int , int , double * , int );
void haar_featlist(int , int , double * , int  , unsigned int * );
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
void MakeIntegralImagesquare(unsigned short int *, unsigned int *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
void qsindex (double  *, int * , int , int );
int eval_haar_subwindow(unsigned int * , unsigned int * , int , int , int  , double  , double , int , struct model , double *);
#ifdef matfx
double * detect_haar(unsigned char * , int  , int , struct model  , int * , double * , double *);
#else
double * detect_haar(unsigned char * , int  , int , struct model  , int * , double *);
#endif
/*------------------------------------------------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{  
    unsigned char *I;
	struct model detector;  
    const int *dimsI ;
    int numdimsI , Tcascade = 0;
    double *D , *Dtemp=NULL , *stat;
	double rect_param_default[40]   = {1 , 1 , 2 , 2 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 0 , 1 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 1 , 0 , 0 , 1 , 1 , -1 , 2 , 2 , 1 , 2 , 2 , 1 , 0 , 1 , 1 , 1};
	double param_default[400]       = {10992.0000,-6.1626,-0.7788,0.0000,76371.0000,24.2334,0.6785,0.0000,4623.0000,-0.6328,-0.5489,0.0000,58198.0000,-4.4234,-0.5018,0.0000,67935.0000,-12.8916,-0.4099,0.0000,19360.0000,-1.7222,0.1743,0.0000,60243.0000,-1.9187,-0.2518,0.0000,3737.0000,-0.9260,0.1791,0.0000,58281.0000,-16.1455,-0.2447,0.0000,13245.0000,-1.6818,0.2183,0.0000,4459.0000,1.7972,0.2043,0.0000,7765.0000,3.0506,0.1665,0.0000,10105.0000,-2.0763,0.1764,0.0000,2301.0000,-2.4221,-0.1526,0.0000,4250.0000,-0.2044,0.1077,0.0000,59328.0000,24.8328,0.2129,0.0000,10127.0000,-2.1996,0.1746,0.0000,65144.0000,-35.6228,-0.2307,0.0000,43255.0000,-0.5288,0.1970,0.0000,57175.0000,-0.2119,0.0597,0.0000,59724.0000,-27.5468,-0.2059,0.0000,13278.0000,-2.1100,0.1895,0.0000,55098.0000,22.4124,0.1913,0.0000,13238.0000,-1.7093,0.1707,0.0000,62386.0000,0.3067,0.1283,0.0000,24039.0000,6.9595,0.1639,0.0000,43211.0000,-0.5982,0.1188,0.0000,62852.0000,9.6709,0.1652,0.0000,43236.0000,-0.6296,0.1530,0.0000,45833.0000,1.7152,0.1974,0.0000,7095.0000,-1.2430,0.1269,0.0000,76347.0000,-27.4002,-0.1801,0.0000,3737.0000,-0.8826,0.1462,0.0000,65143.0000,36.2253,0.1581,0.0000,13160.0000,-2.5302,0.1469,0.0000,4845.0000,0.7053,-0.0690,0.0000,52810.0000,-13.5220,-0.1594,0.0000,43234.0000,0.5907,-0.1420,0.0000,60847.0000,39.2252,0.1563,0.0000,43234.0000,-0.5423,0.1417,0.0000,56659.0000,0.7945,0.1387,0.0000,56930.0000,-1.3875,-0.1496,0.0000,13224.0000,-1.5798,0.1080,0.0000,63154.0000,14.8166,0.1961,0.0000,13162.0000,-2.3354,0.1639,0.0000,10722.0000,0.6559,0.2141,0.0000,7528.0000,1.1026,0.1077,0.0000,4263.0000,0.1324,-0.0485,0.0000,45151.0000,-1.4198,-0.1234,0.0000,7095.0000,1.5141,-0.1367,0.0000,68446.0000,-25.0890,-0.1744,0.0000,43277.0000,-0.5919,0.1564,0.0000,3613.0000,-0.5823,-0.1439,0.0000,5418.0000,3.9535,-0.1502,0.0000,58985.0000,24.7405,0.1754,0.0000,43785.0000,-0.9376,0.1194,0.0000,46582.0000,-5.8589,-0.1286,0.0000,43470.0000,-0.6392,0.1396,0.0000,10262.0000,-2.9209,-0.1251,0.0000,10105.0000,-2.0250,0.0960,0.0000,3555.0000,0.7341,0.1348,0.0000,10115.0000,1.6321,-0.1274,0.0000,76579.0000,-39.8316,-0.1442,0.0000,10228.0000,1.8771,-0.1245,0.0000,57005.0000,2.3937,0.1431,0.0000,43830.0000,-0.7996,0.0652,0.0000,48673.0000,8.8965,0.1181,0.0000,18845.0000,-2.2572,0.0872,0.0000,50225.0000,-1.5850,-0.1181,0.0000,43284.0000,0.5782,-0.1278,0.0000,72000.0000,-8.5961,-0.1282,0.0000,43214.0000,-0.6367,0.1053,0.0000,72559.0000,23.7860,0.1368,0.0000,43792.0000,1.0846,-0.1150,0.0000,56537.0000,-0.1965,-0.1262,0.0000,13421.0000,8.9499,0.0433,0.0000,172.0000,0.5319,-0.0946,0.0000,68220.0000,20.5078,0.1688,0.0000,16105.0000,-1.8842,0.1081,0.0000,79153.0000,5.8776,0.1301,0.0000,19180.0000,2.0606,0.1314,0.0000,13.0000,0.5438,-0.0802,0.0000,67201.0000,-6.6425,-0.1443,0.0000,43210.0000,0.5881,-0.1349,0.0000,65075.0000,-44.1279,-0.1170,0.0000,43214.0000,0.5392,-0.0840,0.0000,139.0000,0.2841,0.1480,0.0000,10209.0000,1.8835,-0.0957,0.0000,44409.0000,-1.0357,-0.1648,0.0000,43210.0000,-0.5252,0.1002,0.0000,47431.0000,6.8252,0.0927,0.0000,10235.0000,-1.3325,0.0795,0.0000,14896.0000,-12.2989,-0.0802,0.0000,1752.0000,-0.8487,-0.1193,0.0000,6964.0000,-1.7033,0.0944,0.0000,64124.0000,32.1583,0.1058,0.0000,43215.0000,-0.6988,0.0997,0.0000,76579.0000,-39.5722,-0.0932,0.0000,43966.0000,1.0216,-0.0926,0.0000,68446.0000,-25.5175,-0.1295,0.0000};
	double cascade_default[16]      = {1, -0.75 , 2 , -0.5 , 3 ,  -0.5 , 4 , -0.25 , 10 , 0 , 20 , 0 , 30 , 0 , 30 , 0};
	double scalingbox_default[3]    = {2 , 1.4 , 1.8};
	double mergingbox_default[3]    = {1/2 , 1/2 , 1/3};

	mxArray *mxtemp;	    
	int i , Ny , Nx  , nD = 0 , tempint , r = 5;
	double *tmp;

#ifdef matfx
	double *fxmat;
#endif

	detector.weaklearner    = 2; 
	detector.epsi           = 0.1;
    detector.nR             = 4;
	detector.cascade_type   = 0;
	detector.postprocessing = 1;
	detector.ny             = 24;
	detector.nx             = 24;
    detector.Ncascade       = 8;
	detector.max_detections = 500;

#ifdef OMP 
    detector.num_threads    = -1;
#endif

	if ((nrhs < 1) || (nrhs > 2) )       
	{
		mexPrintf(
			"\n"
			"\n"
			"Object detector by boosting Haar features (Gentleboosting & Adaboosting)\n"
			"\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"\n"
#ifdef matfx
			"[D , stat , matfx] = detector_haar(I , [model]);\n"
#else
			"[D , stat]  = detector_haar(I , [model]);\n"
#endif
			"\n"
			"\n"
			"Inputs\n"
			"------\n"
			"\n"
			"\n"
			"I                          Input image (Ny x Nx) in UINT8 format.\n"
			"\n" 
			"model                      Trained model structure\n"
			"     weaklearner           Choice of the weak learner used in the training phase (default weaklearner = 2)\n"
			"                           weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R\n"
			"                           weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R\n"
			"                           weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost\n"
			"     param                 Trained parameters matrix (4 x T) in double format. Each row correspond to :\n"
			"                           featureIdx                Feature indexes of the T best weaklearners (1 x T)\n"
			"                           th                        Optimal Threshold parameters (1 x T)\n"
			"                           a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)\n"
			"                           b                         Offset (1 x T)\n"
			"     dimsItraining         Size of the train images used in the mblbp computation, i.e. (ny x nx ) (Default dimsItraining = [24 x 24])\n"
			"     rect_param            Feature's rectangles dictionnary (10 x nR) in double format\n"
			"     F                     Feature's parameters (6 x nF) in UINT32 format\n"
			"     cascade_type          Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade (default cascade_type = 0)\n"
			"     cascade               Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.\n"
			"                           If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :))\n"
			"                           If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))\n"
			"                           cascade(2 , :) reprensent thresholds for each segment\n"
			"     max_detections        Maximum number of raw subwindows detections (default max_detections = 500)\n"
			"     scalingbox            [scale_ini , scale_inc , step_ini] where :\n"
			"                           scale_ini is starting scale factor for each subwindows to apply from the size of trained images (default scale_ini = 2)\n"
			"                           scale_inc is Increment of the scale factor (default scale_inc = 1.4)\n"
			"                           step_ini  is the overlapping subwindows factor such that delta = Round(step_ini*scale_ini*scale_inc^(s)) where s in the number of scaling steps (default step_ini = 2)\n"
			"     postprocessing        Type of postprocessing in order to reduce false alarms (default postprocessing = 1): \n"
			"                           0: no postprocessing, i.e. raw detections, 1: merging if rectangles overlapp more than 25%\n"
			"                           2 : Better Merging detections algorithm with parameters defined by mergingbox\n"
			"     mergingbox            [overlap_same , overlap_diff , step_ini]\n"
			"                           overlap_same is the overlapping factor for merging detections of the same size (first step) (default overlap_same = 1/2)\n"
			"                           overlap_diff is the overlapping factor for merging detections of the different size (second step) (default overlap_diff = 1/2)\n"
			"                           dist_ini is the size fraction of the current windows allowed to merge included subwindows (default dist_ini = 1/3)\n"
			"     max_detections        Maximum number of raw subwindows detections (default max_detections = 500).\n"
#ifdef OMP
			"     num_threads           Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif
			"\n"
			"\n"
			"Outputs\n"
			"-------\n"
			"\n"  
			"D                          Detection results (5 x nD) where nD is the number of detections\n"
			"                           D(1,:) x coordinates of the detections\n"
			"                           D(2,:) y coordinates of the detections\n"
			"                           D(3,:) size of detection windows\n"
			"                           D(4,:) number of merged detection\n"
			"                           D(5,:) detection'values\n"
			"\n"
			"stat                       Number of positives and negatives detection of all scanned subwindows(1 x 2)\n"
#ifdef matfx
			"\n"
			"matfx                      Matrix of raw detections (Ny x Nx)\n"
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
		mxtemp                            = mxGetField( prhs[1] , 0, "weaklearner" );
		if(mxtemp != NULL)
		{	
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 3))
			{						
				detector.weaklearner      = 2;	
			}
			else
			{	
				detector.weaklearner      = tempint;	
			}			
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "epsi" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);

			if(tmp[0] < 0.0 )
			{
				mexPrintf("epsi must be > 0, force to 0.1");	
				detector.epsi             = 0.1;
			}
			else
			{
				detector.epsi             = tmp[0];
			}			
		}

		mxtemp                             = mxGetField( prhs[1], 0, "param" );		
		if(mxtemp != NULL)
		{
			detector.param                 = mxGetPr(mxtemp);
			detector.T                     = mxGetN(mxtemp);
		}
		else
		{
			detector.param                 = (double *)mxMalloc(400*sizeof(double));	
			for(i = 0 ; i < 400 ; i++)
			{
				detector.param[i]          = param_default[i];	
			}	
			detector.T                     = 10;
		}
						
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
		
		mxtemp                              = mxGetField( prhs[1] , 0, "rect_param" );		
		if(mxtemp != NULL)
		{
			if((mxGetM(mxtemp) != 10) || !mxIsDouble(mxtemp) )
			{
				mexErrMsgTxt("rect_param must be (10 x nR) in DOUBLE format");	
			}
			detector.rect_param             = (double *) mxGetData(mxtemp);
			detector.nR                     = mxGetN(mxtemp);	
		}
		else
		{	
			detector.rect_param             = (double *)mxMalloc(40*sizeof(double));
			for(i = 0 ; i < 40 ; i++)
			{		
				detector.rect_param[i]      = rect_param_default[i];	
			}	
		}	
	
		mxtemp                              = mxGetField( prhs[1] , 0, "F" );	
		if(mxtemp != NULL)
		{
			detector.F                      = (unsigned int *)mxGetData(mxtemp);	
			detector.nF                     = mxGetN(mxtemp);
		}
		else
		{
			detector.nF                     = number_haar_features(detector.ny , detector.nx , detector.rect_param , detector.nR);
			detector.F                      = (unsigned int *)mxMalloc(6*detector.nF*sizeof(int));
			haar_featlist(detector.ny , detector.nx , detector.rect_param , detector.nR , detector.F);
		}

		mxtemp                              = mxGetField( prhs[1] , 0, "cascade_type" );
		if(mxtemp != NULL)
		{
			tmp                             = mxGetPr(mxtemp);	
			tempint                         = (int) tmp[0];
			if((tempint < 0) || (tempint > 1))
			{
				detector.cascade_type       = 0;	
			}
			else
			{
				detector.cascade_type       = tempint;	
			}			
		}			
	
		mxtemp                            = mxGetField( prhs[1] , 0, "postprocessing" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 2))
			{								
				detector.postprocessing   = 2;
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
			detector.scalingbox            = mxGetPr(mxtemp);
		}
		else
		{
			detector.scalingbox            = (double *)mxMalloc(3*sizeof(double));
			for(i = 0 ; i < 3 ; i++)
			{
				detector.scalingbox[i]     = scalingbox_default[i];
			}
		}

		mxtemp                             = mxGetField( prhs[1] , 0, "mergingbox" );
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != 3)
			{	
				mexErrMsgTxt("mergingbox must be (1 x 3)");
			}
			detector.mergingbox            = mxGetPr(mxtemp);
		}
		else
		{
			detector.mergingbox             = (double *)mxMalloc(3*sizeof(double));
			for(i = 0 ; i < 3 ; i++)
			{
				detector.mergingbox[i]      = mergingbox_default[i];
			}
		}

		mxtemp                              = mxGetField( prhs[1] , 0, "cascade" );
		if(mxtemp != NULL)
		{
			if(mxGetM(mxtemp) != 2)
			{
				mexErrMsgTxt("cascade must be (2 x Ncascade)");		
			}

			detector.cascade               = mxGetPr(mxtemp );
			detector.Ncascade              = mxGetN(mxtemp);
			for(i = 0 ; i < 2*detector.Ncascade ; i=i+2)
			{
				Tcascade                  += (int) detector.cascade[i];
			}

			if(Tcascade > detector.T)
			{
				mexErrMsgTxt("sum(cascade(1 , :)) <= T");
			}
		}
		else
		{
			detector.cascade              = (double *)mxMalloc(16*sizeof(double));
			for(i = 0 ; i < 16 ; i++)
			{
				detector.cascade[i]       = cascade_default[i];
			}
			detector.Ncascade             = 8;
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "max_detections" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0))
			{								
				detector.max_detections   = 500;
			}
			else
			{
				detector.max_detections   = tempint;	
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
		detector.param                  = (double *)mxMalloc(400*sizeof(double));
		
		for(i = 0 ; i < 400 ; i++)
		{
			detector.param[i]           = param_default[i];	
		}	
		
		detector.T                      = 10;
		detector.rect_param             = (double *)mxMalloc(40*sizeof(double));
		
		for(i = 0 ; i < 40 ; i++)
		{	
			detector.rect_param[i]      = rect_param_default[i];
		}	

		detector.nF                     = number_haar_features(detector.ny , detector.nx , detector.rect_param , detector.nR);		
		
		detector.F                      = (unsigned int *)mxMalloc(6*detector.nF*sizeof(int));
		haar_featlist(detector.ny , detector.nx , detector.rect_param , detector.nR , detector.F);

		detector.scalingbox             = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.scalingbox[i]      = scalingbox_default[i];
		}

		detector.mergingbox             = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.mergingbox[i]      = mergingbox_default[i];
		}

		detector.cascade                = (double *)mxMalloc(16*sizeof(double));
		for(i = 0 ; i < 16 ; i++)
		{
			detector.cascade[i]         = cascade_default[i];
		}
		detector.Ncascade               = 8;
	}
    
    
     /*----------------------- Outputs -------------------------------*/

    plhs[1]                    = mxCreateDoubleMatrix(1 , 2 , mxREAL);
    stat                       = mxGetPr(plhs[1]);

    /*------------------------ Main Call ----------------------------*/


#ifdef matfx

	plhs[2]                    = mxCreateDoubleMatrix(Ny , Nx , mxREAL);
	fxmat                      = mxGetPr(plhs[2]);
	Dtemp                      = detect_haar(I , Ny , Nx  , detector  , &nD , stat , fxmat);
#else
	Dtemp                      = detect_haar(I , Ny , Nx  , detector  , &nD , stat);
#endif
	
    plhs[0]                    = mxCreateDoubleMatrix(r , nD , mxREAL);
    D                          = mxGetPr(plhs[0]);
	
	for(i = 0 ; i < r*nD ; i++)
	{
		D[i]                   = Dtemp[i];	
	}

	/*--------------------------- Free memory -----------------------*/
	
	free(Dtemp);
 
    if ( (nrhs > 1) && !mxIsEmpty(prhs[1]) )
	{	
		if ( mxGetField( prhs[1] , 0 , "param" ) == NULL )		
		{
			mxFree(detector.param);
		}
		if ( mxGetField( prhs[1] , 0 , "rect_param" ) == NULL )	
		{
			mxFree(detector.rect_param);
		}
		if ( mxGetField( prhs[1] , 0 , "F" ) == NULL )	
		{
			mxFree(detector.F);
		}
		if ( mxGetField( prhs[1] , 0 , "scalingbox" ) == NULL )	
		{
			mxFree(detector.scalingbox);
		}
		if ( mxGetField( prhs[1] , 0 , "mergingbox" ) == NULL )	
		{
			mxFree(detector.mergingbox);
		}
		if ( mxGetField( prhs[1] , 0 , "cascade" ) == NULL )	
		{
			mxFree(detector.cascade);
		}
	}
	else
	{
			mxFree(detector.param);
			mxFree(detector.rect_param);
			mxFree(detector.F);
			mxFree(detector.scalingbox);
			mxFree(detector.mergingbox);
			mxFree(detector.cascade);
	}	
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
#ifdef matfx
double * detect_haar(unsigned char *I , int Ny , int Nx  , struct model detector , int *nD , double *stat , double *fxmat )		 
#else
double * detect_haar(unsigned char *I , int Ny , int Nx  , struct model detector  , int *nD , double *stat )
#endif
{   
    double *scalingbox = detector.scalingbox , *mergingbox = detector.mergingbox;
	int ny = detector.ny , nx = detector.nx , postprocessing = detector.postprocessing  , NyNx = Ny*Nx , nys , nxs;
	int Pos_current = detector.max_detections, Pos = 0 , Negs = 0 , ind = 0 , index = 0 , indi , indj , Pos1;
#ifdef OMP 
    int num_threads = detector.num_threads;
#endif
	double tempx , tempy, scale , invscale2, powScaleInc;
	double si , sj , sij;
	unsigned int *II , *Itemp;
	unsigned int *IIsquare;
	unsigned short int *Isquare , tempI;
    double *D , *Draw;
	double *possize;
	int *indexsize;
	
	int i , j , l , m ;
	int yest  , Deltay , Deltax , Ly , Lx , Offsety , Offsetx , Origy , Origx , r = 5;	
	int sizeDataBase = max(nx , ny), halfsizeDataBase = sizeDataBase/2 , current_sizewindow , current_stepwindow, minN = min(Ny,Nx);
	double fx , scale_ini = scalingbox[0] , scale_inc = scalingbox[1] , step_ini = scalingbox[2];
	double overlap_same = mergingbox[0] , overlap_diff = mergingbox[1] , dist_ini = mergingbox[2];
	double dsizeDataBase = (double) sizeDataBase , tmp , nb_detect_total  , nb_detect , nb_detect1 ,  Xinf, Yinf, Xsup , Ysup;

#ifdef matfx
	int indOrigx;
#endif

	II                   = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	IIsquare             = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Itemp                = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Isquare              = (unsigned short int *) malloc(NyNx*sizeof(unsigned short int));	   
    Draw                 = (double *) malloc(r*Pos_current*sizeof(double));


#ifdef OMP 
    num_threads          = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif


	MakeIntegralImage(I , II , Nx , Ny , Itemp);

	for(i = 0 ; i < NyNx ; i++)
	{
		tempI      = I[i];	
		Isquare[i] = tempI*tempI;
	}

	MakeIntegralImagesquare(Isquare , IIsquare , Nx , Ny , Itemp);
		
	current_sizewindow   = halfsizeDataBase*Round(2.0*scale_ini);	
	current_stepwindow   = Round(step_ini*scale_ini);
	powScaleInc          = scale_inc;
	
	while(current_sizewindow <= minN)  
    {
		scale      = (double) (current_sizewindow) / dsizeDataBase;
		invscale2  = 1.0/(scale*scale);		

		nys        = Round(ny*scale) + 1;
		nxs        = Round(nx*scale) + 1;
				
        Deltay     = current_stepwindow;
        Deltax     = current_stepwindow;
						
        Ly         = max(1 , (int) (floor(((Ny - nys)/(double) Deltay))) + 1);
        Offsety    = max(0 , (int) ( floor(Ny - ( (Ly-1)*Deltay + nys + 1)) ));
        
        Lx         = max(1 , (int) (floor(((Nx - nxs)/(double) Deltax))) + 1);
        Offsetx    = max(0 , (int) ( floor(Nx - ( (Lx-1)*Deltax + nxs + 1)) ));
#ifdef OMP 
#ifdef matfx
#pragma omp parallel for default(none) private(m,Origy,yest,fx,index,l,Origx,indOrigx) shared(fxmat,Pos,Negs,Pos_current,Lx,Ly,Offsetx,Offsety,Deltax,Deltay,Draw,II,IIsquare,Ny,r,scale,invscale2,current_sizewindow,detector) 
#else
#pragma omp parallel for default(none) private(m,Origy,yest,fx,index,l,Origx) shared(Pos,Negs,Pos_current,Lx,Ly,Offsetx,Offsety,Deltax,Deltay,Draw,II,IIsquare,Ny,r,scale,invscale2,current_sizewindow,detector) 
#endif
#endif
		/* Shift subwindows */

		for(l = 0 ; l < Lx ; l++) /* Loop shift on x-axis  */
		{
			Origx          = Offsetx + l*Deltax ;
#ifdef matfx
			indOrigx       = Ny*Origx;
#endif		
/*
#ifdef OMP 
#ifdef matfx
#pragma omp parallel for default(none) private(m,Origy,yest,index,fx,indOrigx) shared(fxmat,Pos,Negs,Pos_current,Ly,Origx,Offsety,Deltay,Draw,II,IIsquare,Ny,r,scale,invscale2,current_sizewindow,detector) 
#else
#pragma omp parallel for default(none) private(m,Origy,yest,index) shared(Pos,Negs,Pos_current,Ly,Origx,Offsety,Deltay,Draw,II,IIsquare,Ny,r,scale,invscale2,current_sizewindow,detector) 
#endif
#endif
*/
			for(m = 0 ; m < Ly  ; m++)   /* Loop shift on y-axis  */
			{				
				Origy      = Offsety + m*Deltay ;				
					
				/* Evaluate cascade in (Origy , Origx)*/
				yest                      = eval_haar_subwindow(II,IIsquare,Origy,Origx,Ny,scale,invscale2,current_sizewindow,detector,&fx);
					
#ifdef matfx
				fxmat[Origy + indOrigx]  += fx;
#endif			


				if(yest == 1) /* New raw detection  */		
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
		current_sizewindow        = halfsizeDataBase*Round(2.0*scale_ini*powScaleInc);	
		current_stepwindow        = (int)ceil(step_ini*scale_ini*powScaleInc);
		powScaleInc              *= scale_inc; 	
	}

	if(postprocessing == 0) /* Raw detections */
	{
		nD[0]    = Pos;
		D        = (double *)malloc(Pos*r*sizeof(double));
		indi     = 0;
		for(i = 0 ; i < Pos ; i++)
		{
			D[0 + indi] = (Draw[1 + indi] + 1.0);  /* +1 for matlab */ 
			D[1 + indi] = (Draw[2 + indi] + 1.0);  /* +1 for matlab */ 
			D[2 + indi] = Draw[3 + indi]; 
			D[3 + indi] = Draw[0 + indi];
			D[4 + indi] = Draw[4 + indi]; 		
			indi       += r;
		}
	}
	if(postprocessing == 1)  /* Remove Overlapping False alarms if d(c_i , c_j) < alpha*(R_i+Rj) where c_{i,j} = center of rectangle i,j */
	{	
	     /* Merge detections with equal size and 25 % overlap or more */

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
		
		/* Merge overlapping detections of different size */
		
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
		
		/* Count remaining detections */
		
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
				D[0 + indj] = (Draw[1 + indi] + 1.0);  /* +1 for matlab */ 
				D[1 + indj] = (Draw[2 + indi] + 1.0);  /* +1 for matlab */ 
				D[2 + indj] = Draw[3 + indi]; 
				D[3 + indj] = Draw[0 + indi];
				D[4 + indj] = Draw[4 + indi];				
				indj       += r;
			}
			indi  += r;
		}	
	}

	if(postprocessing == 2)  
	{	
	     /* Merge detections with equal size and 25 % overlap or more */

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

		/* Sort windows size */

		possize                 = (double *) malloc(Pos*sizeof(double));
		indexsize               = (int *) malloc(Pos*sizeof(int));

		for( i = 0 ; i < Pos ; i++)
		{
			possize[i]         = Draw[3 + i*r];
			indexsize[i]       = i;
		}

		qsindex(possize , indexsize , 0 , Pos1);

		
		/* Merge overlapping detections of different size */
				
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
/*
							Draw[1 + indj]  = Draw[1 + indi];
							Draw[2 + indj]  = Draw[2 + indi];
							Draw[3 + indj]  = Draw[3 + indi];
							Draw[4 + indj]  = Draw[4 + indi];
*/
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
		
		/* Count remaining detections */
		
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
				D[0 + indj] = (Draw[1 + indi] + 1.0);  /* +1 for matlab */ 
				D[1 + indj] = (Draw[2 + indi] + 1.0);  /* +1 for matlab */ 
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

	/* Free pointers */
	
	free(Itemp);
	free(II);
	free(Isquare);
	free(IIsquare);
    free(Draw);
	
	stat[0] = (double)Pos;
	stat[1] = (double)Negs;
		
	return D;
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------------------- */
int eval_haar_subwindow(unsigned int *II , unsigned int *IIsquare , int Origy , int Origx , int Ny , double scale , double invscale2 , int current_sizewindow , struct model detector  , double *fx)
{
    double   *param = detector.param , *rect_param = detector.rect_param , *cascade = detector.cascade;
	unsigned int *F = detector.F;
	int Ncascade = detector.Ncascade , weaklearner = detector.weaklearner , cascade_type = detector.cascade_type;
	double epsi  = detector.epsi , sum , sum_total = 0.0, a , b , th , thresh ;
	double var , mean , std;
	int z , c , f , Tc , indc = 0 , indf = 0, idxF ,  x , xr , y , yr , w , wr , h , hr , r , s  , R , indR , coeffw , coeffh;
	int curwin1 = (current_sizewindow - 1) , indbl = Origy + curwin1 , indtl = Origx*Ny , indtr = (Origx + curwin1)*Ny; 
	double ctecurwin = 1.0/(double)(current_sizewindow*current_sizewindow);

	var      = (IIsquare[indbl + indtr] - (IIsquare[Origy + indtr] + IIsquare[indbl + indtl]) + IIsquare[Origy + indtl])*ctecurwin;
	mean     = (II[indbl + indtr] - (II[Origy + indtr] + II[indbl + indtl]) + II[Origy + indtl])*ctecurwin;

/*
    var      = Area(IIsquare , Origx  , Origy  , current_sizewindow , current_sizewindow , Ny)*ctecurwin;
    mean     = Area(II , Origx  , Origy  , current_sizewindow , current_sizewindow , Ny)*ctecurwin;
*/
	std      = sqrt(var - mean*mean);

	if(std == 0.0)
	{
		return 0;
	}

	std      = invscale2/std;
	
	for (c = 0 ; c < Ncascade ; c++)
	{	
		Tc     = (int) cascade[0 + indc];
		thresh = cascade[1 + indc];
		sum    = 0.0;
		
		for (f = 0 ; f < Tc ; f++)
		{	
			idxF  = ((int) param[0 + indf] - 1)*6;
			th    =  param[1 + indf];
			a     =  param[2 + indf];
			b     =  param[3 + indf];
				
			x     =  F[1 + idxF];
			y     =  F[2 + idxF];
			w     =  F[3 + idxF];
			h     =  F[4 + idxF];

			indR  =  F[5 + idxF];
			R     = (int) rect_param[3 + indR];

			z     = 0;
			for (r = 0 ; r < R ; r++)
			{	
				coeffw  = w/(int)rect_param[1 + indR];			
				coeffh  = h/(int)rect_param[2 + indR];
				xr      = Round(scale*(x + (coeffw*(int)rect_param[5 + indR]))) + Origx;
				yr      = Round(scale*(y + (coeffh*(int)rect_param[6 + indR]))) + Origy;
				wr      = Round(scale*(coeffw*(int)(rect_param[7 + indR])));
				hr      = Round(scale*(coeffh*(int)(rect_param[8 + indR])));
				s       = (int)rect_param[9 + indR];
				z      += s*Area(II , xr  , yr  , wr , hr , Ny);				
				indR   += 10;
			}				
			if(weaklearner == 0)			
			{
				sum    += (a*( (z*std) > th ) + b);	
			}
			if(weaklearner == 1)
			{	
				sum    += ((2.0/(1.0 + exp(-2.0*epsi*(th*(z*std) + b)))) - 1.0);	
			}
			if(weaklearner == 2)
			{
				sum    += a*sign((z*std) - th);
			}						
			indf      += 4;			
		}
		
		sum_total     += sum;

		if((sum_total < thresh) && (cascade_type == 1))
		{
			fx[0]     = sum_total;
			return 0;
		}		
		else if((sum < thresh) && (cascade_type == 0))
		{
			fx[0]     = sum;
			return 0;				
		}			
		indc      += 2; 
	}
	if(cascade_type == 1)
	{
		fx[0]     = sum_total;	
	}
	else if(cascade_type == 0)
	{
		fx[0]     = sum;	
	}
	return 1;
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
	int Y , X;
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
void MakeIntegralImagesquare(unsigned short int *pIn, unsigned int *pOut, int iXmax, int iYmax , unsigned int *pTemp)
{
	/* Variable declaration */

	int x , y , indx  = 0;
		
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
int Round(double x)
{
	return (int)(x + 0.5);
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
