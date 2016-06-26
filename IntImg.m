function [A] = IntImg(img)
% Purpose:
%   Calculate the integral image of a window
% 
% Author: Quang Nguyen
%

A = cumsum(cumsum(double(img)),2);