clear; close all
% cd 'C:\Users\mypc\Desktop\New folder\Data'
% addpath 'D:\Research Project\fdtool_release\fdtool_release'
load('model_hmblbp_R4.mat');

min_detect                      = 2; %2
model.postprocessing            = 2;
% model.scalingbox                = [2 , 1.35 , 1.75];
% model.mergingbox                = [1/2 , 1/2 , 0.8];

%vid = videoinput('winvideo' , 1 , 'RGB24_640x480');
    vid = videoinput('winvideo' , 1 , 'YUY2_320x240');
    % My  own Commands
    set(vid,'ReturnedColorSpace', 'rgb');
    % End of my command
    preview(vid); 
    fig1 = figure(1);
    set(fig1,'doublebuffer','on');
    while(1)
        t1   = tic;
        aa   = getsnapshot(vid);
        pos  = detector_mlhmslbp_spyr(rgb2gray(aa) , model);
        image(aa);
        hold on
        for i=1:size(pos,2)
             if(pos(4 , i) >= min_detect)
               rectangle('Position', [pos(1,i),pos(2,i),pos(3,i),pos(3,i)], 'EdgeColor', [0,1,0], 'linewidth', 2);
               %x = pos(1,i); y = pos(2,i); size = pos(3,i);
               %cropIm = aa(y:(y+size),x:(x+size),:);
               %imwrite(cropIm,'data16.jpg','jpg');
             end
        end
        hold off
        t2 = toc(t1);
        title(sprintf('Fps 5 %6.3f      (Press CRTL+C to stop)' , 1/t2));
        drawnow;
    end