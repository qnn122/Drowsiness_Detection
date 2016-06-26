function DrowsinessDetectionGUI()
% Purpose:
% This file creates a graphical user interface for drowsiness monitoring.
% In short, the function does following tasks:
%   1. Preparing data
%   2. Create GUI
%   3. Monitor awareness's state of the user, which consists of:
%       3.1. Capture image
%       3.2. Detect user's face
%       3.3. Localize the eyes
%       3.4. Monitor eyes states (open/close)
%       3.5. Monitor drowsiness level
%  Task 3 is performed by the sub-function DetectDrowsiness
% 
% Author : Quang Nguyen
%

% =============== Preparing data =============================
if exist('vid','var'), delete(vid); end
close all, clear all
% Load necessary local variables
load coord2;    coord = coord2; 
load AB;        AB = AB;
load haarPara;  haarPara = haarPara;
load modelRF;   modelRF = modelRF;
addpath fdtool_release\fdtool_release
addpath RF_Class_C

% Load and set Haar model up
load('model_hmblbp_R4.mat'); model = model;
min_detect                      = 2; %2
model.postprocessing            = 2;

% Set up camera
NumberFrameDisplayPerSecond = 20;
% vid = videoinput('winvideo',1,'RGB24_320x240');
% vid = videoinput('winvideo',1,'YUY2_320x240');
vid = videoinput('winvideo',1, CameraSetup())
set(vid,'ReturnedColorSpace', 'rgb');
set(vid, 'FramesPerTrigger', 1); % capture 1 frame when triggered
set(vid, 'TriggerRepeat', Inf);
triggerconfig(vid, 'Manual');

% Create figure
h_fig = figure('name', 'Drowsiness Detection Tool');
set(h_fig,'doublebuffer','on');
set(h_fig,'CloseRequestFcn', @CloseReq);

% Create 2 axes to contain video feed
ax1 = axes('Parent', h_fig,...
            'Position', [.03 .2 .45 .68]);
ax2 = axes('Parent', h_fig,...
            'Position', [.52 .2 .45 .68]);

% Create start button which calls the startbCallback function
startb = uicontrol('Parent', h_fig,...
            'Units', 'normalized',...
            'Position', [.6 .02 .16 .08],...
            'Style', 'pushbutton',...
            'String', 'START',...
            'FontSize', 18,...
            'Callback', @startbCallback);

% Create stop button which calls the stopbCallback function
stopb = uicontrol('Parent', h_fig, ...
            'Units', 'normalized', ...
            'Position', [.8 .02 .16 .08],...
            'Style', 'pushbutton',...
            'String', 'STOP',...
            'FontSize', 18,...
            'Callback', @stopbCallback);

% Create popup menu for switching face/eye display
htextpop = uicontrol('Style', 'text',...
            'String', 'Select display '' scope and mode', ...
            'Units', 'normalized', 'Position', [.02, .96, .4, .03]);
hpopup = uicontrol('Style', 'popupmenu',...
            'String', {'Face', 'Eye: after global thresholding', 'Eye: after image dilate'},...
            'Units', 'normalized',...
            'Position', [.02 .86 .17 .08],...
            'Callback',{@popup_menu_Callback});
popup.cond1 = 1; popup.cond2 = 0; popup.cond3 = 0;
set(hpopup, 'UserData', popup);
h_chbox0 = uicontrol(h_fig, 'Style', 'checkbox',...
            'String', 'Display 5 regions',...
            'Units', 'normalized',...
            'Position', [.2 .91 .1 .03],...
            'Value', 0,...
            'Callback',{@chbox0_Callback});
checkbox.Disp5reg = 0;
set(h_chbox0, 'UserData', checkbox);

% Create a panel for display analysis info
h_panel1 = uipanel(h_fig, 'Title','Analyzing signal',...
            'Units', 'normalized',...
            'Position', [.02, .02, .3, .15]);
h_chbox1 = uicontrol(h_panel1, 'Style', 'checkbox',...
            'String', 'Weighted Average',...
            'Units', 'normalized',...
            'Position', [.05 .65 .6 .2],...
            'Value', 1,...
            'Callback',{@chbox1_Callback});
checkbox.DispWeightedAve = 1;
set(h_chbox1, 'UserData', checkbox);
h_chbox2 = uicontrol(h_panel1, 'Style', 'checkbox',...
            'String', 'Adaptive Thresholding',...
            'Units', 'normalized',...
            'Position', [.05 .2 .6 .2],...
            'Value', 1,...
            'Callback',{@chbox2_Callback});
checkbox.DispAdaptiveThresh = 1;
set(h_chbox2, 'UserData', checkbox);
h_chbox3 = uicontrol(h_panel1, 'Style', 'checkbox',...
            'String', 'Open/Close state',...
            'Units', 'normalized',...
            'Position', [.42 .65 .5 .2],...
            'Value', 0,...
            'Callback',{@chbox3_Callback});
checkbox.DispOpenClose = 0;
set(h_chbox3, 'UserData', checkbox);
h_chbox4 = uicontrol(h_panel1, 'Style', 'checkbox',...
            'String', 'Warning Level',...
            'Units', 'normalized',...
            'Position', [.42 .2 .5 .2],...
            'Value', 0,...
            'Callback',{@chbox4_Callback});
checkbox.DispWarning = 0;
set(h_chbox4, 'UserData', checkbox);

% Create a dummy static textbox to check what's going on
h_text = uicontrol(h_fig, 'Style', 'text',...
            'String','System status', 'Units', 'normalized',... 
            'Position', [.52 .9 .2 .03]);


% Create timer that calls the dispim fctn every 0.05 sec
t = timer('TimerFcn', {@DetectDrowsiness, checkbox, popup}, ...
            'Period', 1/NumberFrameDisplayPerSecond, ...
            'executionMode', 'fixedRate','BusyMode','drop');   

            
% ------------------------------------------------------------------------
    function DetectDrowsiness(hobj, event, whatever, whatever2)
    % DetectDrowsiness function does all image processing and 
    % displays the result

    % Declare variables
    persistent framelim Wd warn_win indx lastval lastthresh laststate lastdrowsy
    persistent aa aa2 aa3 pos
    persistent posiFeat haarFeat 
    persistent classlabel
    persistent RE RE_cen REreg LE LE_cen LEreg
    persistent h_plot1 h_plot2 h_plot3 h_plot4 h_plotlabel
    persistent im1 im2 im12 im22 value1 value2 value thresh T1 T2 flag flag2
    persistent data state
    persistent a h_BG h_RE h_LE h_NOSE h_MOUTH
    persistent drowsyLev
    DEBUG = 0;
    framelim = 100; % limited number of frames
    Wd = 20;        % Sliding window for calculating ADAPTIVE THRESHOLD
    warn_win = 15;  % Sliding window for calculating WARNING LEVEL (DROWSINEES LEV)
    
    % Initialize persistent variables
    if isempty(indx),       indx = 1;               end
    if isempty(lastval),    lastval = nan;          end
    if isempty(lastthresh), lastthresh = nan;       end
    if isempty(laststate),  laststate = 0;          end
    if isempty(lastdrowsy), lastdrowsy = nan;       end
    if isempty(flag),       flag = 1;               end
    if isempty(flag2),      flag2 = 1;               end
    
    if isempty(data),       data = zeros(1,framelim);    end
    if isempty(state),      state = zeros(1,framelim);   end
    
    trigger(vid);               % Trigger vid to capture image
    aa   = getsnapshot(vid);    % Capture image
    
    % ================== FACE DETECTION =================================
    pos  = detector_mlhmslbp_spyr(rgb2gray(aa) , model);
    if popup.cond1
        imshow(aa, 'parent', ax1); 
    end
    hold on;

    % Announce the condition of the system
    if indx < Wd && flag
        set(h_text, 'String', 'Initializing...');
    else
        set(h_text, 'String', 'Start analyzing');
        flag = 0;   % Make sure that when indx become 1 again, 
                    % the system status still remains 'Start analyzing'
    end
    
    % Initialize threshold
    if flag2
        thresh = 0.2;
        flag2 = 0;
    end
        
    % =================== EYE MONITORING ==================================
    for i=1:size(pos,2)
         if(pos(4 , i) >= min_detect)
            aa2 = rgb2gray(aa);
            x = pos(1,i) - 5; y = pos(2,i) - 5; width =  1.1*pos(3,i);
            if popup.cond1
                rectangle('Position', [x,y,width,width], 'EdgeColor', ...
                            [0,1,0], 'linewidth', 2, 'parent', ax1);
            end
            
            % ============= FACIAL REGIONS INDENTIFICATION ================
            if DEBUG, disp('reach 1'); end
            coord2 = floor(coord/128*width);
            coord2(:,1) = coord2(:,1) + y; coord2(:,2) = coord2(:,2) + x;
            
            aa3 = histeq(imresize(imcrop(aa2,[x,y,width,width]),[128 128]));
            posiFeat = CreatePosiFeat_mex(double(aa3), coord, AB);
            haarFeat = CreateHaarFeat_mex(IntImg(double(aa3)), coord, haarPara);
            
            classlabel = classRF_predict([posiFeat haarFeat],modelRF) + 1;
            
            % ---------------- Display 5 regions -------------------------
            if checkbox.Disp5reg
                a = [coord2 classlabel]; 
                h_BG = plot(a(a(:,3)==1,2),a(a(:,3)==1,1),'co',...
                            'MarkerFaceColor','c'); % background
                h_RE = plot(a(a(:,3)==2,2),a(a(:,3)==2,1),'ro',...
                            'MarkerFaceColor','r'); % right eye
                h_LE = plot(a(a(:,3)==3,2),a(a(:,3)==3,1),'yo',...
                            'MarkerFaceColor','y'); % left eye
                h_NOSE = plot(a(a(:,3)==4,2),a(a(:,3)==4,1),'bo',...
                            'MarkerFaceColor','b'); % nose
                h_MOUTH = plot(a(a(:,3)==5,2),a(a(:,3)==5,1),'go',...
                            'MarkerFaceColor','g'); % mouth
                set(h_BG, 'parent', ax1);
                set(h_RE, 'parent', ax1);
                set(h_LE, 'parent', ax1);
                set(h_NOSE, 'parent', ax1);
                set(h_MOUTH, 'parent', ax1);
            end
            % -----------------------------------------------------------
            
            % Determine the eyes' positions
            RE = coord2(classlabel==2,:);
            RE_cen = round(mean(RE));
            LE = coord2(classlabel==3,:);
            LE_cen = round(mean(LE));
         
            % ========== EXTRACT AND PROCESS EYES' REGIONS ================
            if DEBUG, disp('reach 2'); end
            % Extract LE region
            W = (LE_cen(1)-round(width*0.1)):(LE_cen(1)+round(width*0.1));
            L = (LE_cen(2)-round(width*0.13)):(LE_cen(2)+round(width*0.1));
            LEreg = aa2(W ,L);
            % Extract RE region
            W2 = (RE_cen(1)-round(width*0.1)):(RE_cen(1)+round(width*0.1));
            L2 = (RE_cen(2)-round(width*0.13)):(RE_cen(2)+round(width*0.1));
            REreg = aa2(W2, L2);
            rectangle('Position', [x,y,width,width], 'EdgeColor', [0,1,0],...
                        'linewidth', 2, 'parent', ax1);
            % Plot eyes' regions
            if popup.cond1
                % Left one
                rectangle('Position', [L(1) W(1) length(L) length(W)],...
                            'EdgeColor', [1,0,0], 'linewidth',2,'parent',ax1);
                % Right one
                rectangle('Position', [L2(1) W2(1) length(L2) length(W2)],...
                            'EdgeColor', [1,0,0], 'linewidth',2,'parent',ax1);           
            end
            
            % Processing LE region
            im1 = LEreg<graythresh(LEreg)*0.3*width;
            im2 = imdilate(im1, strel('disk',2));
            value1 = mean(mean(im2,2)*3,1);
            if popup.cond2 
                imshow(im1, 'parent', ax1);
            end
            if popup.cond3
                imshow(im2, 'parent', ax1);
            end
          
            % Processing RE region
            im12 = REreg<graythresh(REreg)*0.3*width;
            im22 = imdilate(im12, strel('disk',2));
            value2 = mean(mean(im22,2)*3,1);
            
            % Plot signal from eyes
            xlim = 50;
            set(gca, 'xlim', [floor(indx/xlim)*xlim, floor(indx/xlim)*xlim + xlim]);
            set(gca, 'ylim', [-0.2 1.5]);
            value = mean([value1 value2]);
            % ------------------------------------------------------------
            if checkbox.DispWeightedAve && indx~=1
                h_plot3 = plot([indx-1 indx],[lastval value],'b.-'); grid on;
                set(h_plot3, 'parent', ax2);
            end
            % -----------------------------------------------------------
            
            lastval = value; % update
            data(indx) = value;
            
            % ================= ADAPTIVE THRESHOLDING =====================
            if DEBUG, disp('reach 3'); end
            if indx == 1
                if DEBUG, disp('reach 3_2'); end
                thresh = ( max([data(framelim-Wd+1:framelim), data(1:indx)]) + ...
                            min([data(framelim-Wd+1:framelim), data(1:indx)]) )/2;

            elseif (indx>1)&&(indx<=Wd)
                if DEBUG, disp('reach 3_3'); end
                T1 = mean(data( [find(data(1:(indx-1))>thresh), ...
                                 find(data(framelim-(Wd-indx):framelim)>thresh) + framelim-(Wd-indx+1)] ));
                T2 = mean(data( [find(data(1:(indx-1))<thresh), ...
                                 find(data(framelim-(Wd-indx):framelim)<thresh) + framelim-(Wd-indx+1)] ));
                if DEBUG, disp('reach 3_4'); end
                if isnan(T1)|isnan(T2)|T1>1|T2>1
                    T1 = max(data( [(1:(indx-1)), (framelim-(Wd-indx)):framelim] )); 
                    T2 = min(data( [(1:(indx-1)), (framelim-(Wd-indx)):framelim] )); 
                end
                thresh = mean([T1 T2])*(1-0.15);
            elseif indx > Wd
                T1 = mean(data(find(data(indx-Wd:(indx-1))>thresh)+(indx-Wd)));
                T2 = mean(data(find(data(indx-Wd:(indx-1))<thresh)+(indx-Wd)));
                if isnan(T1)|isnan(T2)||T1>1||T2>1
                    T1 = max(data((indx-Wd):(indx-1)));
                    T2 = min(data((indx-Wd):(indx-1)));
                end
                thresh = mean([T1 T2])*(1-0.2);
            end

            % ---------- Display Adaptive Threshold ----------------------
            if checkbox.DispAdaptiveThresh 
                h_plot4 = plot([indx-1 indx],[lastthresh thresh]); 
                set(h_plot4, 'parent', ax2, 'Color', [1 .5 0], ...
                    'LineStyle', '--', 'LineWidth', 4);
            end
            % ------------------------------------------------------------
            lastthresh = thresh; % update
            
            % Plot binary states of eyes
            if value < thresh
                state(indx) = 1;
            end
            % --------- Disp Open/Clos state -----------------------------
            if checkbox.DispOpenClose && indx~=1
                stem([indx-1 indx],[state(indx-1) state(indx)],'Color','g');
            end
            % ------------------------------------------------------------
            
            % Reset the 2 axes 
            if indx == framelim
                indx = 1;
                % Re-setting up ax1
                cla(ax1, 'reset'); grid(ax1);
                set(ax1, 'xlim', [floor(indx/xlim)*xlim, floor(indx/xlim)*xlim + xlim]);
                set(ax1, 'ylim', [0 1.5]);      
                % Re-setting up ax2
                cla(ax2, 'reset'); grid(ax2);
                set(ax2, 'xlim', [floor(indx/xlim)*xlim, floor(indx/xlim)*xlim + xlim]);
                set(ax2, 'ylim', [-0.2 1.2]);             
            end
            
            % =========== DROSINESS DETECTION RULES =======================
            if DEBUG, disp('reach 4'); end
            if indx == 1
                state(2:(framelim-warn_win+2)) = 0;
                drowsyLev = sum(state(framelim-warn_win+1:framelim))/warn_win;
            elseif (indx>1)&&(indx<warn_win)
                drowsyLev = sum([state(1:indx-1) ...
                                state(framelim-(warn_win-indx-1):framelim)])/warn_win;
            elseif indx == warn_win
                drowsyLev = sum(state((indx-warn_win+1):indx))/warn_win;
                state((framelim-warn_win+3):framelim) = 0;
            elseif indx > warn_win
                drowsyLev = sum(state((indx-warn_win+1):indx))/warn_win;
            end
            % Display the results
            if DEBUG, disp('reach 5'); end
            if checkbox.DispWarning && indx~=1
                plot([indx-1 indx],[lastdrowsy drowsyLev],'r.-','MarkerSize',17);
            end
            
            if drowsyLev > 0.55
                % Raise warning when the drowsiness level exceeds threshold
                beep on 
%                 text(floor(indx/xlim)*xlim+0.2,0.8,'\color{red}DROWSINESS DETECTED');
            end
            if DEBUG, disp('reach 6'); end
         end % detecting criteria
    end % for
    hold off
    % Update indx
    indx = indx + 1;
    
    drawnow;
    end % DetecDrowsiness() sub-function


% ---------------------------------------------------------------
% Figure Closing Request
    function CloseReq(hobj, event)
        selection = questdlg('Close this Figure?', ...
            'Close Request Function', ...
            'Yes', 'No', 'Yes');
        switch selection,
            case 'Yes',
                %stop(t); delete(t);
                stop(vid); delete(vid);
                delete(gcf);
            case 'No'
            return
        end
    end
% ---------------------------------------------------------------
% CallBack of START BUTTON: start image aquisition
    function startbCallback(hobj, event)
        start(vid);
        start(t);
    end

% ---------------------------------------------------------------
% Stop video streaming
    function stopbCallback(hobj, event)
        stop(t);
        stop(vid);
    end

% --------------------------------------------------------------
% Display 5 facial regions
    function chbox0_Callback(hObject, event)
        checkbox.Disp5reg = get(hObject, 'Value');
    end

% ---------------------------------------------------------------
    function chbox1_Callback(hObject, event)
        checkbox.DispWeightedAve = get(hObject, 'Value');
    end

% ----------------------------------------------------------------
    function chbox2_Callback(hObject, event)
        checkbox.DispAdaptiveThresh = get(hObject, 'Value');
    end

% ---------------------------------------------------------------
    function chbox3_Callback(hObject, event)
        checkbox.DispOpenClose = get(hObject, 'Value');
    end

% ----------------------------------------------------------------
    function chbox4_Callback(hObject, event)
        checkbox.DispWarning = get(hObject, 'Value');
    end

% ----------------------------------------------------------------
    function popup_menu_Callback(hObject, event)
        val = get(hObject, 'Value');
        switch val
            case 1
                popup.cond1 = 1; popup.cond2 = 0; popup.cond3 = 0;
            case 2
                popup.cond1 = 0; popup.cond2 = 1; popup.cond3 = 0;
            case 3
                popup.cond1 = 0; popup.cond2 = 0; popup.cond3 = 1;
        end
    end
end % main function
