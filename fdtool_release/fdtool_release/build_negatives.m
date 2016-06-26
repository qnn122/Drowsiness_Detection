function build_negatives(options)

%
%   Usage
%  ------
%
%  build_negatives(options);
%
%
%  Example
%  -------
%  options.keywords        = {'trees' , 'landsc/ape','desk office' , 'large office','book' , 'kitchen' , 'corridor','whiteboard' , 'seafront' , 'parking building' , 'gallery' , 'shops'};
%  options.Nimneg          = 20;
%  options.pathneg         = fullfile(pwd , 'images' , 'train' , 'negatives');
%  build_negatives(options);
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009


if(nargin < 1)
    options.keywords             = {'trees' , 'wall' , 'landscape','office desk','book'};
    options.Nimneg               = 500;
    options.pathneg              = fullfile(pwd , 'images' , 'train' , 'negatives');
end

if(~any(strcmp(fieldnames(options) , 'Nimneg')))
    options.Nimneg               = 500;
end
if(~any(strcmp(fieldnames(options) , 'pathneg')))
    options.pathneg              = fullfile(pwd , 'images' , 'train' , 'negatives');
end
if(~exist(options.pathneg , 'dir'))
    mkdir(options.pathneg);
end


nb_keywords                      = length(options.keywords);
imneg                            = cell(options.Nimneg,nb_keywords);

for i = 1:nb_keywords
    temp     = ieJPGSearch(options.keywords{i});
    for j = 1:options.Nimneg
        if(~isempty(temp{j}))
            try
                A                        = imread(temp{j});
                [PATHSTR,NAME,EXT,VERSN] = fileparts(temp{j});
                imwrite(A, fullfile(options.pathneg , [NAME,EXT]) , 'jpeg');
                imneg{j,i}               = temp{j};
                disp(['saving ' temp{j}])
                pause(0.2)
            end
        end
    end
end