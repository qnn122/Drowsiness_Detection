function h = plot_rectangle(D , color)


%  Overplot rectangles associated with face detections 
%
%  Usage
%  ------
% 
%  h = plot_rectangle(D , [color])
%
%  Inputs
%  ------
%   
%  D                 Detected rectangles matrix (5 x nR) (see detector_haar or detector_mblbp functions)
%  color             Color of the rectangles (default color = 'g')
% 
%  Output
%  ------
%
%  h                 Handle on the plot
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009
%
%

if(nargin < 2)
    color  = 'g';
end

[v , nR]   = size(D);
xy         = repmat(D([1,2] , :) , 7 , 1);
l          =  D(3 , :);
wh         = [zeros(3 , nR)  ;  l    ;    zeros(1 , nR) ; l    ;   l ; l   ;   l ; l    ; l  ; zeros(3 , nR)];


Rect       = xy + wh;

h          = plot(Rect(1:2:end-1 , :),Rect(2:2:end , :) , color , 'linewidth' , 3);