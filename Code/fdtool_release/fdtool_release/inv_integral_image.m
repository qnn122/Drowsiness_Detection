function I = inv_integral_image(II , type)

%  Retrieve the original Image I from the Integral Image II
%
%  Usage
%  ------
% 
%  I = inv_integral_image(II , [type])
%
%  Inputs
%  ------
%   
%  II                Integral Image's database (Ny x Nx x N)
%  type              type of integration :
%                    type = 1, i.e. II = cumsum(cumsum(I,1),2)
%                    type = 2, i.e. II = cumsum(cumsum(I,2),1) (default)
% 
%
%  Example
%  -------
%
%  load viola_24x24
%  I  = X(: , : , 110);
%  II = image_integral_standard(I);
%  clear I
%  I  = inv_integral_image(II);
%  imagesc(I)
%  colormap(gray)
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009
%


if(nargin < 2)

    type = 2;

end

if(type)

    III = [II(: , 1) , II(: , 2:end ) - II(: , 1:end-1)];

    I   = [III(1 , :) ; III(2:end , :) - III(1:end-1 , :)];

else

    III = [II(1  , :) ; II(2:end , :) - II(1:end-1 , :)];

    I   = [III(: , 1) , III(: , 2:end) - III(: , 1:end-1)];

end