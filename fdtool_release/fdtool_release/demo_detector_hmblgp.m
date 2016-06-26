clear,close all

webcam_driver                   = 1; %1 = vcapg2, 2 = matlab imaq toolbox

load('model_hmblgp_R4.mat');

min_detect                      = 2; %2
model.postprocessing            = 2;
% model.scalingbox                = [2 , 1.35 , 1.75];
% model.mergingbox                = [1/2 , 1/2 , 0.8];

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
        pos  = detector_mlhmslgp_spyr(rgb2gray(aa) ,model);   
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
        pos  = detector_mlhmslgp_spyr(rgb2gray(aa) , model);
        image(aa);
        hold on
        h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g');
        hold off
        t2 = toc(t1);
        title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));
        drawnow;
    end
end