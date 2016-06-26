
clear,close all

model_choice                    = 1;
webcam_driver                   = 1; %1 = vcapg2, 2 = matlab imaq toolbox



if(model_choice == 1)
    
%     load('temp_model3.mat');
%     min_detect                  = 35;
    
   
  
    load('model_detector_haar_24x24.mat');
    model.postprocessing        = 2;
    min_detect                  = 2; %2
    model.cascade               = [1 , 2 , 3 , 4 , 10 , 20 , 30 , 30 ; -0.75 ,-0.6 , -0.5, -0.25,  0 ,  0 , 0 , 0];

end
if(model_choice == 2)
   
    load('model_detector_haar_24x24_nP4.mat');
    min_detect                  = 2;
    model.cascade               = [2 , 8 , 10 , 20 ,  20 , 20; -0.5 , -0.5 , -0.25, -0.25 , 0 , 0 ];
%    cascade                     = [1 , 2 , 3 , 4 , 10 , 20 ; -1 ,-0.75 , -0.5, -0.5,  0 ,  0];
    
end

model.scalingbox                = [2 , 1.35 , 1.75];
model.mergingbox                = [1/2 , 1/2 , 0.8];
%model.mergingbox                = [1/2 , 1/2 , 1/3];


if(ispc)  
    if(isempty(ver('imaq')))
        webcam_driver                   = 1;
    end
else
    if(~isempty(ver('imaq')))   
        webcam_driver                   = 2;
    else
        error('No Image Acquisition Toolbox installed')
    end
       
end

if(webcam_driver == 1)
    
    aa                             = vcapg2(0,3);
    
    fig1 = figure(1);
    set(fig1 , 'doublebuffer' , 'on' , 'renderer' , 'zbuffer');
    drawnow;
    
    while(1)
        t1   = tic;        
        aa   = vcapg2(0,0);
        pos  = detector_haar(rgb2gray(aa) ,model);
        
        image(aa);
        hold on
        h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g' );
        hold off
        t2   = toc(t1);
        
        title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));
        
        drawnow;
    end 
end

if(webcam_driver == 2)
    
    vid = videoinput('winvideo' , 1 , 'RGB24_640x480');
    preview(vid);
    
    fig1 = figure(1);
    set(fig1,'doublebuffer','on');
    while(1)
        t1   = tic;
        aa   = getsnapshot(vid);
        pos  = detector_haar(rgb2gray(aa) , model);
        
        image(aa);
        hold on
        h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g');
        hold off
        t2 = toc(t1);
        title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));      
        drawnow;
    end
end