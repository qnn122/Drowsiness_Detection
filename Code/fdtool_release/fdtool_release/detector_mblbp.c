/*

  Object detector based on MBLBP features and trained by boosting algorithms (Gentleboosting & Adaboosting) 

  Usage
  ------

  [D , stat , [matfx]] = detector_mblbp(I , [model]);

  Inputs
  -------

  I                                     Input image (Ny x Nx) in UINT8 format
  
  model                                 Trained model structure

             weaklearner                Choice of the weak learner used in the training phase (default weaklearner = 0)
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
			                            weaklearner = 2 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a))|), where h(x;(th,a)) = a*sign(z - th)  in [-1,1] for discrete adaboost
             param                      Trained parameters matrix (4 x T) in double format. Each row correspond to :
                                        featureIdx                Feature indexes of the T best weaklearners (1 x T)
			                            th                        Optimal Threshold parameters (1 x T)
			                            a                         WeakLearner's weights (1 x T) in R (at = ct*pt, where pt = polarity)
			                            b                         Offset (1 x T)
			 dimsItraining              Size of the train images used in the mblbp computation, i.e. (ny x nx ) (default dimsItraining = [24 , 24])
			 F                          Feature's parameters (5 x nF) in UINT32 format
			 map                        Mapping of the lbp used in the MBLPB computation (1 x 256) in UINT8 format (default map = (0 : 255))
             cascade_type               Type of cascade structure : 0 for coventional cascade, 1 for multi-exit cascade
             cascade                    Cascade parameters (2 x Ncascade) where cascade(1 , :) represents Entrance/Exit nodes.
                                        If cascade_type = 0, i.e. coventional cascade, Entrance nodes are [1 , cumsum(cascade(1 , 1:end-1))+1] and exit nodes are cumsum(cascade(1 , :)) 
										If cascade_type = 1, i.e. multi-exit cascade, Entrance node is 1, exit nodes are cumsum(cascade(1 , :))
										cascade(2 , :) reprensents thresholds for each segment
             postprocessing             Type of postprocessing in order to reduce false alarms (default postprocessing = 1): 
			                            0: no postprocessing, i.e. raw detections, 1: merging if rectangles overlapp more than 25%
										2 : Better Merging detections algorithm with parameters defined by mergingbox
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

			num_threads                 Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)


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


  mex  -output detector_mblbp.dll detector_mblbp.c

  mex  -f mexopts_intel10.bat -output detector_mblbp.dll detector_mblbp.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -f mexopts_intel10.bat -output detector_mblbp.dll detector_mblbp.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"

  or with the matfx option

  mex  -Dmatfx -f mexopts_intel10.bat -output detector_mblbp.dll detector_mblbp.c

  If OMP directive is added, OpenMP support for multicore computation

  mex -v -DOMP -Dmatfx -f mexopts_intel10.bat -output detector_mblbp.dll detector_mblbp.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"



  Example 1
  ---------


  close all
  load model_detector_mblbp_24x24_4.mat

  model.cascade_type       = 0;
  model.cascade            = [5 , 10 , 15 , 29 ; -0.1 , -0.25 ,  0 , 0 ];
  model.scalingbox         = [1.7 , 1.2 , 1.5];
  model.mergingbox         = [1/2 , 1/2 , 0.8];

  min_detect               = 3;
  
  I                        = rgb2gray(imread('class57.jpg'));
 
  tic,[D , stat]           = detector_mblbp(I , model);,toc

  rect                     = [D(1 , :)' , D(2 , :)' , D(3 , :)' , D(3 , :)'];
  [z,its]                  = size(D);

  %close all
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

  figure, imshow(I);
  hold on;
  for i=1:its
  if(D(4,i)>=min_detect)
    rectangle('Position',[D(1,i),D(2,i),D(3,i),D(3,i)],'Edgecolor',[0,1,0],'LineWidth',2);
    text(D(1,i)+D(3,i)/2,D(2,i)+D(3,i)/2,num2str(D(4,i)),'FontSize',15,'Color',[0,0,0],'BackgroundColor',[1,1,1]);

  end
  end
  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))




  Example 2
  ---------

  clear
  clear vcapg

  close all
  load ('model_detector_mblbp_24x24_4.mat');
%  mblbp                    = load ('model_detector_mblbp_24x24_wl0_ct1.mat');
%  mblbp                    = load ('model_detector_mblbp_24x24_wl0_ct1_1.mat');


  model.cascade_type        = 1;
  model.cascade             = [2 , 8 , 10 , 20 , 25 ; -0*0.25 , -0*0.25 , 0*0.25 ,  0 , 0 ];
 % cascade                  = [3 , 7 , 10 , 10 , 20 , 50 ; -0.5 , -0.25 , -0.25 ,  -0.25 , 0 , 0 ]; 
 % cascade                  = [3 4 2 1;0 0 0 0]; 
 % cascade                  = [6 2 3 2 1 3 14 3 4;0 0 0 0 0 0 0 0 0];

  model.scalingbox         = [2 , 1.4 , 2];
  min_detect               = 3;

  aa                       = vcapg2(0,2);

  figure(1);set(1,'doublebuffer','on');
  while(1)
    aa   = vcapg2(0,0);
    pos  = detector_mblbp(rgb2gray(aa) , model);

    imagesc(aa);
    hold on
    for i=1:size(pos,2)
	   if(pos(4 , i) >= min_detect)
        rectangle('Position', [pos(1,i),pos(2,i),pos(3,i),pos(3,i)], 'EdgeColor', [0,1,0], 'linewidth', 2);
		end
    end
    hold off

    drawnow;
  end


  Example 3
  ---------

  clear all,close all
      
  aa                          = vcapg2(0,2);

  load ('model_detector_mblbp_24x24_4.mat');;

  model.cascade               = [2 , 8 , 10 , 20 , 25 ; -0*0.125 , 1*0.125 , 2*0.125 ,  1*0.125 , 0 ];
  %model.cascade               = [1 , 2 , 4 , 8 , 10 , 20 ; -0.5 , -2*0.125 , -1*0.125 , 0*0.125 , 0*0.125 ,  1*0.125  ];
  model.scalingbox            = [2 , 1.4 , 1.75];

  min_detect                  = 3;
  fig1 = figure(1);
  set(fig1,'doublebuffer','on');
  while(1)
    t1   = tic;
    aa   = vcapg2(0,0);
    pos  =  detector_mblbp(rgb2gray(aa) , model);

    image(aa);
    hold on
    h = plot_rectangle(pos(: , find(pos(4 , :) >=min_detect)) , 'g');    
	hold off
	t2 = toc(t1);
	title(sprintf('Fps = %6.3f' , 1/t2));

    drawnow;
  end



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
#define sign(a)    ((a) >= (0) ? (1.0) : (-1.0))
 
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

struct model
{
	int             weaklearner;
	double          epsi;
	double          *param;
	int             T;
	double          *dimsItraining;
	int             ny;
	int             nx;
	unsigned int    *F;
	int             nF;
	unsigned char  *map;
	int             cascade_type;
	int             postprocessing;
	double         *scalingbox;
	double         *cascade;
	int             Ncascade;
	int             max_detections;
	double         *mergingbox;
#ifdef OMP 
    int            num_threads;
#endif
};

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

int Round(double );
int number_mblbp_features(int , int );
void mblbp_featlist(int  , int , unsigned int *);
void MakeIntegralImage(unsigned char *, unsigned int *, int , int , unsigned int *);
unsigned int Area(unsigned int * , int , int , int , int , int );
int eval_mblbp_subwindow(unsigned int * , int , int , int , double  , struct model , double *);
void qsindex (double  *, int * , int , int );

#ifdef matfx
double * detect_mblbp(unsigned char * , int  , int  , struct model  ,  int * , double * , double * );
#else
double * detect_mblbp(unsigned char * , int  , int  , struct model  ,  int * , double * );
#endif

/*-------------------------------------------------------------------------------------------------------------- */

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	unsigned char *I;
	struct model detector;
	const int *dimsI ;
	int numdimsI , Tcascade = 0;
	double *D , *Dtemp=NULL , *stat;

	double param_default[400]       = {4608.0000,240.0000, 1.1750,-0.6868,8396.0000, 7.0000,-1.0774, 0.6278,4717.0000,12.0000,-0.9667, 0.5909,6130.0000,223.5000, 0.7644,-0.3870,4776.0000,127.0000,-0.7657, 0.2433,5027.0000, 1.5000,-0.8168, 0.6111,2399.0000,192.0000, 0.6640,-0.3146,986.0000,240.0000, 0.6726,-0.2658,5289.0000,63.0000,-0.5540, 0.2837,1002.0000,227.5000, 0.6053,-0.2320,2571.0000,161.0000, 0.5391,-0.2587,6774.0000,46.0000,-0.5479, 0.3235,5389.0000,220.0000, 0.5034,-0.1697,986.0000,48.0000, 0.7172,-0.6259,3067.0000,234.0000,-0.7112, 0.0879,822.0000,228.0000,-0.6506, 0.0896,2411.0000,190.0000, 0.4595,-0.2182,185.0000,32.0000, 0.5747,-0.4633,4239.0000,143.0000,-0.5122, 0.1384,1309.0000,183.0000, 0.4818,-0.2120,4131.0000,64.0000,-0.4564, 0.2447,1145.0000,119.0000, 0.4988,-0.3431,2274.0000,195.0000, 0.4599,-0.2104,2753.0000,237.0000,-0.6033, 0.1039,2805.0000,24.0000, 0.6233,-0.5368,1611.0000,56.0000, 0.5743,-0.4798,5769.0000,225.0000,-0.5952, 0.0857,715.0000,128.0000, 0.4319,-0.2546,7914.0000,233.0000,-0.6488, 0.0804,2896.0000,241.0000, 0.4893,-0.1312,6555.0000,225.0000,-0.7045, 0.0771,2450.0000,14.0000, 0.6013,-0.5252,96.0000, 2.0000, 0.7296,-0.6644,1223.0000, 8.0000, 0.6591,-0.5927,308.0000, 2.0000, 0.7811,-0.7270,136.0000,207.0000,-0.5026, 0.0877,349.0000,237.0000,-0.5503, 0.0791,2164.0000,31.0000, 0.5243,-0.4402,4100.0000,157.0000, 0.3848,-0.1920,4078.0000,224.0000,-0.5150, 0.1033,2330.0000,60.0000, 0.4320,-0.3157,768.0000,252.0000,-0.8387, 0.0613,751.0000, 8.0000, 0.5779,-0.5144,1000.0000, 6.0000, 0.7844,-0.7250,74.0000, 8.0000, 0.5917,-0.5229,892.0000,191.0000, 0.3999,-0.1474,7986.0000,64.0000,-0.3796, 0.2177,2667.0000,48.0000, 0.4651,-0.3600,6691.0000,191.0000, 0.4077,-0.1707,2153.0000,63.0000,-0.3979, 0.2135,7633.0000,56.0000, 0.4288,-0.3232,6274.0000,68.0000,-0.3929, 0.2427,3818.0000,63.0000,-0.3822, 0.1872,586.0000,239.0000, 0.4054,-0.1366,3901.0000, 8.0000, 0.5937,-0.5174,238.0000,229.0000, 0.4361,-0.1072,2825.0000,95.0000,-0.3792, 0.2139,2031.0000,193.0000, 0.4069,-0.1689,2577.0000,143.0000,-0.4686, 0.1123,326.0000,55.0000, 0.4383,-0.3149,4119.0000,24.0000, 0.5279,-0.4414,928.0000,246.0000,-0.5931, 0.0655,2411.0000,227.0000,-0.5046, 0.0793,2334.0000,10.0000, 0.5918,-0.5350,2362.0000,11.0000, 0.5816,-0.5155,424.0000,25.0000, 0.4660,-0.3772,3088.0000,211.5000,-0.4360, 0.0976,2408.0000,26.0000, 0.5043,-0.4183,316.0000,223.0000, 0.3862,-0.1356,263.0000, 6.0000, 0.5788,-0.5087,2157.0000,227.0000,-0.5186, 0.0681,6581.0000,12.0000, 0.5558,-0.4945,1197.0000,227.0000,-0.4703, 0.0799,402.0000, 0.0000, 0.7701,-0.7340,2814.0000,12.0000, 0.5069,-0.4388,5812.0000,246.0000, 0.3962,-0.1081,197.0000,31.0000,-0.3743, 0.2453,2135.0000,12.0000, 0.5345,-0.4676,608.0000,249.0000,-0.7812, 0.0486,74.0000,62.0000,-0.3577, 0.2071,74.0000,193.0000, 0.3861,-0.1436,6545.0000,253.0000, 0.5373,-0.0608,302.0000,60.0000, 0.4138,-0.2837,2785.0000,192.0000, 0.3753,-0.1568,3153.0000,247.0000,-0.8017, 0.0470,7534.0000,14.0000,-0.4093, 0.2770,619.0000, 6.0000, 0.6074,-0.5501,2690.0000,252.0000,-0.6929, 0.0480,6657.0000,227.0000,-0.5275, 0.0568,5959.0000, 0.0000,-0.5938, 0.5327,163.0000,56.0000, 0.4318,-0.3468,3418.0000, 4.0000,-0.5357, 0.4684,1336.0000,246.0000, 0.4439,-0.0912,3609.0000, 0.0000,-0.5008, 0.4336,3013.0000,126.0000, 0.3753,-0.2499,1959.0000,244.0000, 0.5115,-0.0858,2703.0000,251.0000, 0.5093,-0.0679,5731.0000,253.0000,-0.7091, 0.0533,157.0000,131.0000,-0.3844, 0.1128,136.0000,249.0000,-0.6834, 0.0491};
	double cascade_default[16]      = {1, -0.75 , 2 , -0.5 , 3 ,  -0.5 , 4 , -0.25 , 10 , 0 , 20 , 0 , 30 , 0 , 30 , 0};
	double scalingbox_default[3]    = {2 , 1.4 , 1.8};
	double mergingbox_default[3]    = {1/2 , 1/2 , 0.8};

	mxArray *mxtemp;	    
	int i , Ny , Nx , powN  = 256 , nD = 0 , tempint , r = 5;
	double *tmp;

#ifdef matfx
	double *fxmat;
#endif
	detector.weaklearner    = 0; 
	detector.epsi           = 0.1;
	detector.cascade_type   = 0;
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
			"Object detector on MBLBP features and trained by boosting algorithms (Gentleboosting & Adaboosting) \n"
			"\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"\n"
#ifdef matfx
			"[D , stat , matfx] = detector_mblbp(I , [model]);\n"
#else
			"[D , stat]  = detector_mblbp(I , [model]);\n"
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
			"     F                     Feature's parameters (5 x nF) in UINT32 format\n"
			"     map                   Mapping of the lbp used in the MBLPB computation (1 x 256) in UINT8 format (default map = (0 : 255))\n"
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
		return;	}

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
				detector.weaklearner      = 0;	
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

		mxtemp                             = mxGetField( prhs[1] , 0, "F" );
		if(mxtemp != NULL)
		{	
			detector.F                     = (unsigned int *)mxGetData(mxtemp);	
			detector.nF                    = mxGetN(mxtemp);	
		}
		else
		{
			detector.nF                    = number_mblbp_features(detector.ny , detector.nx);
			detector.F                     = (unsigned int *)mxMalloc(5*detector.nF*sizeof(int));	
			mblbp_featlist(detector.ny , detector.nx , detector.F);	
		}

		mxtemp                             = mxGetField( prhs[1] , 0, "map" );
		if(mxtemp != NULL)
		{		
			if(mxGetN(mxtemp) != powN)
			{		
				mexErrMsgTxt("map must be (1 x 256) in UINT8 format");	
			}
			detector.map                   = (unsigned char *) mxGetData(mxtemp);
		}
		else
		{
			detector.map                   = (unsigned char *)mxMalloc(powN*sizeof(unsigned char));

			for(i = 0 ; i < powN ; i++)
			{		
				detector.map[i]            = (unsigned char) i;	
			}	
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade_type" );
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if((tempint < 0) || (tempint > 1))
			{
				detector.cascade_type     = 0;	
			}
			else
			{
				detector.cascade_type     = tempint;	
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
			detector.scalingbox               = mxGetPr(mxtemp);
		}
		else
		{
			detector.scalingbox             = (double *)mxMalloc(3*sizeof(double));

			for(i = 0 ; i < 3 ; i++)
			{
				detector.scalingbox[i]       = scalingbox_default[i];
			}
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "mergingbox" );
		if(mxtemp != NULL)
		{
			if(mxGetN(mxtemp) != 3)
			{	
				mexErrMsgTxt("mergingbox must be (1 x 3)");
			}
			detector.mergingbox               = mxGetPr(mxtemp);
		}
		else
		{
			detector.mergingbox             = (double *)mxMalloc(3*sizeof(double));

			for(i = 0 ; i < 3 ; i++)
			{
				detector.mergingbox[i]       = mergingbox_default[i];
			}
		}

		mxtemp                            = mxGetField( prhs[1] , 0, "cascade" );
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
				Tcascade         += (int) detector.cascade[i];
			}
			if(Tcascade > detector.T)
			{
				mexErrMsgTxt("sum(cascade(1 , :)) <= T");
			}
		}
		else
		{
			detector.cascade                = (double *)mxMalloc(16*sizeof(double));
			for(i = 0 ; i < 16 ; i++)
			{
				detector.cascade[i]             = cascade_default[i];
			}
			detector.Ncascade               = 8;
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
		detector.param                 = (double *)mxMalloc(400*sizeof(double));

		for(i = 0 ; i < 400 ; i++)
		{		
			detector.param[i]          = param_default[i];	
		}	
		detector.T                     = 10;

		detector.nF                    = number_mblbp_features(detector.ny , detector.nx);			
		detector.F                     = (unsigned int *)mxMalloc(5*detector.nF*sizeof(int));
		mblbp_featlist(detector.ny , detector.nx , detector.F);

		detector.map                   = (unsigned char *)mxMalloc(powN*sizeof(unsigned char));

		for(i = 0 ; i < powN ; i++)
		{
			detector.map[i]            = (unsigned char) i;	
		}

		detector.scalingbox            = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.scalingbox[i]     = scalingbox_default[i];
		}

		detector.mergingbox             = (double *)mxMalloc(3*sizeof(double));
		for(i = 0 ; i < 3 ; i++)
		{
			detector.mergingbox[i]        = mergingbox_default[i];
		}

		detector.cascade               = (double *)mxMalloc(16*sizeof(double));
		for(i = 0 ; i < 16 ; i++)
		{
			detector.cascade[i]        = cascade_default[i];
		}
		detector.Ncascade              = 8;
	}

	plhs[1]                    = mxCreateDoubleMatrix(1 , 2 , mxREAL);
	stat                       = mxGetPr(plhs[1]);


	/*------------------------ Main Call ----------------------------*/

#ifdef matfx
	plhs[2]                    = mxCreateDoubleMatrix(Ny , Nx , mxREAL);
	fxmat                      = mxGetPr(plhs[2]);
	Dtemp                      = detect_mblbp(I , Ny , Nx  , detector  , &nD , stat , fxmat);
#else
	Dtemp                      = detect_mblbp(I , Ny , Nx  , detector  , &nD , stat);
#endif

	/*----------------------- Outputs -------------------------------*/

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
		if ( mxGetField( prhs[1] , 0 , "F" ) == NULL )		
		{
			mxFree(detector.F);
		}
		if ( mxGetField( prhs[1] , 0 , "map" ) == NULL )	
		{
			mxFree(detector.map);
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
		mxFree(detector.F);
		mxFree(detector.map);
		mxFree(detector.scalingbox);
		mxFree(detector.mergingbox);
		mxFree(detector.cascade);
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */

#ifdef matfx
double * detect_mblbp(unsigned char *I , int Ny , int Nx  , struct model detector , int *nD , double *stat , double *fxmat)
#else
double * detect_mblbp(unsigned char *I , int Ny , int Nx  , struct model detector , int *nD , double *stat )
#endif
{
	double *scalingbox = detector.scalingbox , *mergingbox = detector.mergingbox;
	double *D , *Draw;
	unsigned int *II , *Itemp;
	double *possize;
	int *indexsize;

	double scale_ini = scalingbox[0] , scale_inc = scalingbox[1] , step_ini = scalingbox[2];
	double overlap_same = mergingbox[0] , overlap_diff = mergingbox[1] , dist_ini = mergingbox[2];
	double si , sj , sij;

	int ny = detector.ny , nx = detector.nx , NyNx = Ny*Nx , postprocessing = detector.postprocessing;
	int sizeDataBase = max(nx , ny), halfsizeDataBase = sizeDataBase/2 , current_sizewindow , current_stepwindow;
	int Pos_current = detector.max_detections, Pos=0 , Pos1, Negs=0 , ind = 0 , index = 0 , indi , indj,minN = min(Ny,Nx);
#ifdef OMP 
    int num_threads = detector.num_threads;
#endif
	int i , j , l , m ;
	int yest , Deltay , Deltax , Ly , Lx , Offsety , Offsetx , Origy , Origx , nys, nxs , r = 5;

	double tempx , tempy, scale , powScaleInc , dsizeDataBase = (double) sizeDataBase;
	double tmp , nb_detect_total , nb_detect , nb_detect1, Xinf, Yinf, Xsup, Ysup , fx;

#ifdef matfx
	int indOrigx;
#endif

	II                              = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Itemp                           = (unsigned int *) malloc(NyNx*sizeof(unsigned int));
	Draw                            = (double *) malloc(r*Pos_current*sizeof(double));

#ifdef OMP 
    num_threads                     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif


	MakeIntegralImage(I , II , Nx , Ny , Itemp);

	current_sizewindow              = halfsizeDataBase*Round(2.0*scale_ini);	
	current_stepwindow              = Round(step_ini*scale_ini);
	powScaleInc                     = scale_inc;


	while(current_sizewindow <= minN)  
	{
		scale      = (double) (current_sizewindow) / dsizeDataBase ;

		nys        = Round(ny*scale) + 1;  /* + 1 since max(3*round(scale*h)) = round(scale*y) + 1 whatever scale factor */
		nxs        = Round(nx*scale) + 1;  /* + 1 since max(3*round(scale*w)) = round(scale*w) + 1 whatever scale factor */

		Deltay     = current_stepwindow ;
		Deltax     = current_stepwindow ;

		Ly         = max(1 , (int) (floor(((Ny - nys)/(double) Deltay))) + 1);
		Offsety    = max(0 , (int)( floor(Ny - ( (Ly-1)*Deltay + nys + 1)) ));

		Lx         = max(1 , (int) (floor(((Nx - nxs)/(double) Deltax))) + 1);
		Offsetx    = max(0 , (int)( floor(Nx - ( (Lx-1)*Deltax + nxs + 1)) ));

#ifdef OMP 
#ifdef matfx
#pragma omp parallel for default(none) private(m,Origy,yest,fx,index,l,Origx,indOrigx) shared(fxmat,Pos,Negs,Pos_current,Lx,Ly,Offsetx,Offsety,Deltax,Deltay,Draw,II,Ny,r,scale,current_sizewindow,detector) 
#else
#pragma omp parallel for default(none) private(m,Origy,yest,fx,index,l,Origx) shared(Pos,Negs,Pos_current,Lx,Ly,Offsetx,Offsety,Deltax,Deltay,Draw,II,Ny,r,scale,current_sizewindow,detector) 
#endif
#endif

		/* Shift subwindows */

		for(l = 0 ; l < Lx ; l++) /* Loop shift on x-axis */
		{
			Origx          = Offsetx + l*Deltax ;

#ifdef matfx
			indOrigx       = Ny*Origx;
#endif

			for(m = 0 ; m < Ly ; m++)   /* Loop shift on y-axis  */
			{
				Origy      = Offsety + m*Deltay ;				

				/* Evaluate cascade in (Origy , Origx)*/

				yest                      = eval_mblbp_subwindow(II , Origy , Origx , Ny , scale , detector , &fx);

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
		D        = (double *) malloc(Pos*r*sizeof(double));
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
	free(Draw);

	stat[0] = (double)Pos;
	stat[1] = (double)Negs;
	return D;
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
int eval_mblbp_subwindow(unsigned int *I , int Origy , int Origx , int Ny , double scale , struct model detector  , double *fx)
{
	double   *param = detector.param , *cascade = detector.cascade;
	unsigned int *F = detector.F;
	unsigned char *map = detector.map;
	int Ncascade = detector.Ncascade, weaklearner = detector.weaklearner , cascade_type = detector.cascade_type;
	double epsi = detector.epsi;
	int xc , yc , xnw , ynw , xse , yse , w , h;
	unsigned int Ac ;
	unsigned char valF , z;
	double sum , sum_total = 0.0, a , b , th , thresh;
	int c , f , Tc , indc = 0 , indf = 0, idxF;

	for (c = 0 ; c < Ncascade ; c++)
	{
		Tc     = (int) cascade[0 + indc];		
		thresh = cascade[1 + indc];
		sum    = 0.0;
		for (f = 0 ; f < Tc ; f++)
		{
			idxF  = ((int) param[0 + indf] - 1)*5;
			th    = param[1 + indf];
			a     = param[2 + indf];
			b     = param[3 + indf];

			xc    = Round(scale*(F[1 + idxF])) + Origx;
			yc    = Round(scale*(F[2 + idxF])) + Origy;

			w     = Round(scale*F[3 + idxF]);
			h     = Round(scale*F[4 + idxF]);

			xnw   = xc - w;
			ynw   = yc - h;
			xse   = xc + w;
			yse   = yc + h;

			Ac    = Area(I , xc  , yc  , w , h , Ny);

			valF  = 0;
			if(Area(I , xnw , ynw , w , h , Ny) > Ac)
			{
				valF |= 0x01;
			}
			if(Area(I , xc  , ynw , w , h , Ny) > Ac)
			{
				valF |= 0x02;
			}
			if(Area(I , xse , ynw , w , h , Ny) > Ac)
			{
				valF |= 0x04;				
			}
			if(Area(I , xse , yc  , w , h , Ny) > Ac)
			{
				valF |= 0x08;		
			}
			if(Area(I , xse , yse , w , h , Ny) > Ac)
			{
				valF |= 0x10;
			}
			if(Area(I , xc  , yse , w , h , Ny) > Ac)
			{
				valF |= 0x20;
			}
			if(Area(I , xnw , yse , w , h , Ny) > Ac)
			{
				valF |= 0x40;
			}
			if(Area(I , xnw , yc  , w , h , Ny) > Ac)
			{
				valF |= 0x80;
			}
			z        = map[valF];

			if(weaklearner == 0)			
			{
				sum    += (a*( z > th ) + b);	
			}
			else if(weaklearner == 1)
			{
				sum    += ((2.0/(1.0 + exp(-2.0*epsi*(th*z + b)))) - 1.0);	
			}
			else if(weaklearner == 2)
			{
				sum    += a*sign(z - th);	
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
	if(cascade_type == 1 )
	{
		fx[0]     = sum_total;	
	}
	else if(cascade_type == 0 )
	{
		fx[0]     = sum;	
	}
	return 1;
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
}/*----------------------------------------------------------------------------------------------------------------------------------------------*/
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
