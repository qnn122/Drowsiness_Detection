function options = perf_dr_fa(options , model)
%
% Compute detection rate versus number of false alarms
% for face detectors
%
%
%
%  clear, close all
%  options.pathfacesimg  = 'C:\utilisateurs\SeBy\Matlab\MIT-CMU-frontal-face-set-4-Timo\MIT+CMU frontal face set 4 Timo';
%  options.gt_file       = 'C:\utilisateurs\SeBy\Matlab\MIT-CMU-frontal-face-set-4-Timo\FacePara5points.txt';
%  options.dataset       = 1; 
%  options.refbox        = 0;
%  options.overlapping   = 0.5;
%  options.minwratio     = 0.4;
%  options.maxwratio     = 2;
%  options.threshmin     = -1;
%  options.threshmax     = 1;
%  options.nbins         = 50;
%  options.detector      = 0;
%  options.preview       = 1;
% 
%  load ('C:\utilisateurs\SeBy\Matlab\fdtool\model_detector_haar_24x24.mat')
%  model.cascade         = [1 , 2 , 3 , 4 , 10 , 20 , 30 , 30 ; -0.75 , -0.6 , -0.6, -0.5,  -0.4 ,  -0.3 , -0.2 , 0];
%%  model.cascade         = [1 , 2 , 3 , 4 , 10 , 20 , 30 , 30 ; -0.75 ,-0.6 , -0.5, -0.25,  -0.2 ,  0 , 0 , 0];
%  model.scalingbox      = [1.15 , 1.30 , 1.6];
%  model.mergingbox      = [1/2 , 1/2 , 1/3];
%  model.min_detect      = 1;
%  model.postprocessing  = 1;
%  
%  options               = perf_dr_fa(options , model);
% 
%  figure(1)
%  h                     = semilogx(options.fppw_mean , 1-options.tpr_mean);
%  h                     = semilogx(sort(options.fppw_mean) , sort(1-options.tpr_mean , 'descend'));
%  set(h , 'linewidth' , 2)
%  axis([min(options.fppw_mean)*0.95 , max(options.fppw_mean)*1.05 , -0.05 , 1.05])
%  xlabel('False Positive Per Window (FFFW)')
%  ylabel('Miss rate')
%  h                     = title('MIT-CMU-frontal');
%  set(h , 'fontsize' , 13);
% 
%  thresh = options.threshmin:(options.threshmax-options.threshmin)/(options.nbins-1):options.threshmax;
%  figure(2)
%  plot(options.fpp(end:-1:1) , options.tpp(end:-1:1) , 'linewidth' , 2)
%  xlabel('False Positives')
% % addTopXAxis('expression', 'thresh''', 'xLabStr', '\lambda')
%  ylabel('True detection rate')
%  axis([min(options.fpp)*0.9 , max(options.fpp)*1.005 , -0.05 , 1.05])
%  grid on
%  h                     = title('MIT-CMU-frontal');
%  set(h , 'fontsize' , 13);
%   
%  [options.nbtruedetperim ; options.tpperim(1 , :)]
%

[pathgt, gt_file, gt_ext] = fileparts(options.gt_file);
fid                       = fopen(fullfile(pathgt , [gt_file , gt_ext]));
if(options.preview)
    fig = figure(1);
    set(fig,'doublebuffer','on');
    colormap(gray)
end

if(options.dataset == 1)
    
    C                   = textscan(fid,'%s%d%d%d%d%d%d%d%d%d%d%d%d');
    options.detperfile  = C{1};
    yle                 = double(C{2})';
    xle                 = double(C{3})';
    yre                 = double(C{4})';
    xre                 = double(C{5})';
    xlm                 = double(C{8})';
    ylm                 = double(C{9})';
    xrm                 = double(C{12})';
    yrm                 = double(C{13})';
    
    
%    w                   = sqrt((xle-xre).^2 + (yle-yre).^2)/0.4;
    w                   = sqrt((xle-xre).^2 + (yle-yre).^2)/0.47;

    h                   = 1.42*w;
%    x                   = xle - 0.3*w;
    x                   = xle - 0.52*w;    
    y                   = (yle + yre)./2 - 0.5*h;
    
    options.facebox     = [y ; x ; h ; w];
    
    wmax                = 1.15*w;
    hmax                = 1.15*h;
    xmax                = x -(wmax - w)./2;
    ymax                = y -(hmax - h)./2;
    
    options.facemax     = [ymax ; xmax ; hmax ; wmax];
    
    xmin                = xle - 0.11*w;
    ymin                = (yle + yre)/2 - 0.06*h;
    wmin                = (xre + 0.11*w) - xmin;
    hmin                = (ylm + yrm)/2+0.07*h - ymin;
    
    options.facemin     = [ ymin; xmin ; hmin ; wmin];
    
    
    if(options.refbox == 0)
        refbox        = options.facebox;
    elseif(options.refbox == 1)
        refbox        = options.facemax;
    elseif(options.refbox == 2)
        refbox        = options.facemin;
    end
    
    
    options.image_names     = unique(options.detperfile);
    options.nb_images       = length(options.image_names);
    options.nbtruedetperim  = zeros(1 , options.nb_images);
    options.nbestdetperim   = zeros(options.nbins , options.nb_images);
    options.tpperim         = zeros(options.nbins , options.nb_images);
    options.fpperim         = zeros(options.nbins , options.nb_images);
    options.totalscan       = zeros(options.nbins , options.nb_images);
    options.deltathresh     = (options.threshmax-options.threshmin)/(options.nbins - 1);
    
    if(options.detector == 3)
       bias                 = model.w(end); 
    end
    
    for i = 1 : options.nb_images
        
        pathi                     = fullfile(options.pathfacesimg , options.image_names{i});
        I                         = imread(pathi);
        [Ny , Nx , Nz]            = size(I);
        if(Nz == 3)
            I                     = rgb2gray(I);
        end
        
        indexi                    = find(strcmp(options.image_names{i} , options.detperfile));
        options.nbtruedetperim(i) = length(indexi);
        
        for b = 1:options.nbins
                        
            if(options.detector == 0)
                thresh                 = options.threshmin + (b-1)*options.deltathresh;
                model.cascade(2, end)  = thresh;          
                [D , stat]             = detector_haar(I , model);
                D                      = D(: , D(4 , :)>=model.min_detect);
            elseif(options.detector == 1)
                thresh                 = options.threshmin + (b-1)*options.deltathresh;
                model.cascade(2, end)  = thresh;          
                [D , stat]             = detector_mblbp(I , model);
                D                      = D(: , D(4 , :)>=model.min_detect);
            elseif(options.detector == 2)
                thresh                 = options.threshmin + (b-1)*options.deltathresh;
                model.w(end)           = bias + thresh;
                [D , stat]             = detector_mlhmslbp_spyr(I , model);
                D                      = D(: , D(4 , :)>=model.min_detect);
            end
            
            options.totalscan(b , i)   = sum(stat);
            options.nbestdetperim(b,i) = size(D , 2);
            
            indexdet                   = (1:options.nbestdetperim(b,i));
            indextrue                  = [];
                       
            for j = 1 : options.nbtruedetperim(i)
  
                refboxj   = refbox(: , indexi(j));
                x1t       = refboxj(1);
                y1t       = refboxj(2);
                wt        = refboxj(3);
                ht        = refboxj(4);
                x2t       = x1t + wt;
                y2t       = y1t + ht;
                wtmin     = options.minwratio*wt;
                wtmax     = options.maxwratio*wt;
                htmin     = options.minwratio*ht;
                htmax     = options.maxwratio*ht;
               
                areat     = (x2t-x1t+1).*(y2t-y1t+1);
                
                last      = length(indexdet);
                suppress  = [];
                overmax   = -inf;
                for pos = 1:last
                    if(indexdet(pos)~= 0)
                        posi    = pos;
                        
                        x1e     = D(1,posi);
                        y1e     = D(2,posi);
                        we      = D(3,posi);
                        he      = D(3,posi);
                        x2e     = x1e + we;
                        y2e     = y1e + he;
                        
                        xx1     = max(x1t , x1e);
                        yy1     = max(y1t , y1e);
                        xx2     = min(x2t , x2e);
                        yy2     = min(y2t , y2e);
                        ww      = xx2-xx1+1;
                        hh      = yy2-yy1+1;
                        
                        if ((ww > 0) && (hh > 0) && (we > wtmin) && (he > htmin) && (we < wtmax) && (he < htmax))
                            % compute overlap
                            o = ww * hh / areat;
                            if ((o > options.overlapping) && (o > overmax))
                                suppress             = pos;
                                overmax              = o;
                            end
                        end
                    end
                end
                if(~isempty(suppress))
                    options.tpperim(b,i) = options.tpperim(b,i)+1;
                    indexdet(suppress)   = 0;
                    indextrue            = [indextrue , suppress];                   
                 end
            end
            options.fpperim(b,i)       = options.nbestdetperim(b,i) - options.tpperim(b,i);
            
            if(options.preview)            
                indexfalsea            = 1:size(D , 2);
                indexfalsea(indextrue) = [];
                imagesc(I)
                hold on
                plot_rectangle(D(: , indexfalsea) , 'r');
                plot_rectangle(D(: , indextrue)   , 'g');
                plot_rectangle(refbox(: , indexi) , 'b');
                colormap(gray)
                h = text(D(1 , :) , D(2 , :) , int2str((1:size(D,2))'));
                set(h , 'fontsize' , 12 ,'color' , [1 1 0]);
                hold off
                h=title(sprintf('image (%d/%d), true detec (%d/%d), false alarms = %d, detected windows = %d, scanned windows = %d, \\lambda = %4.2f' , i , options.nb_images , options.tpperim(b,i) , options.nbtruedetperim(i) , options.fpperim(b,i) , options.nbestdetperim(b,i) , options.totalscan(b , i) , thresh));
                set(h , 'fontsize' , 13 );
                drawnow
%                pause(0.01)
            end
        end
    end
    
    options.tpr_mean  = mean(options.tpperim./options.nbtruedetperim(ones(options.nbins , 1) , :) , 2);
    options.fppw_mean = mean(options.tpperim./options.totalscan , 2);
    
    options.tpp       =  sum(options.tpperim , 2)/sum(options.nbtruedetperim);
    options.fpp       =  sum(options.fpperim , 2);

    
end
fclose(fid);




