conf.path           = pwd;
conf.negdir         = {'images' , 'train' , 'negatives'};
conf.posdir         = {'images' , 'train' , 'positives'};

cd(conf.path)

% --------------------------------------------------------------------
%                Create empty dir if absent
% --------------------------------------------------------------------

if (~exist(fullfile(conf.path,conf.negdir{:}), 'dir'))
    fprintf('Creating negatives dir\n');
    mkdir(fullfile(conf.path,conf.negdir{:}));
    drawnow
end

if (~exist(fullfile(conf.path,conf.posdir{:}), 'dir'))
    fprintf('Creating positives dir\n');
    mkdir(fullfile(conf.path,conf.posdir{:}));
    drawnow
end


% --------------------------------------------------------------------
%                Unzip Negatives picts
% --------------------------------------------------------------------

fprintf('Unzipping negatives picts ...\n');
drawnow
unzip('negatives.zip' , fullfile(conf.path,conf.negdir{:}));


% --------------------------------------------------------------------
%                Compile mex-files
% --------------------------------------------------------------------


C  = computer;


try
    fprintf('Compile mex files ...\n');
    drawnow

%     options.config_file = 'mexopts_intel10.bat';
%     options.ext         = 'dll';
%     options.useOMP      = 1;
%     options.userBLAS    = '"C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\C++\10.1.013\IA32\lib\libiomp5md.lib"';
%     mexme_fdt(options);


%     options.config_file = 'mexopts_intel11_64.bat';
%     options.ext         = 'mexw64';
%     options.useOMP      = 1;
%     options.userBLAS    = '"C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_core.lib" "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_intel_lp64.lib" "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_intel_thread.lib"';
%     mexme_fdt(options);
  
  mexme_fdt;
  
    if(ispc)
        if(strcmp(C , 'PCWIN'))
            unzip('vcapg2w32.zip' , conf.path);
            fprintf('unzipping vcapg2 ...\n');
            drawnow            
        elseif(strcmp(C , 'PCWIN64'))
            unzip('vcapg2w64.zip' , conf.path);
            fprintf('unzipping vcapg2 ...\n');
            drawnow
            
        end
    end    
catch ME
    if(ispc)
        if(strcmp(C , 'PCWIN'))
            fprintf('Failed to compile mex-files, unzip precompiled mex32\n') ;
            unzip('mexw32.zip' , conf.path);
            unzip('vcapg2w32.zip' , conf.path);
        elseif(strcmp(C , 'PCWIN64'))
            fprintf('Failed to compile mex-files, unzip precompiled mex64\n') ;
            unzip('mexw64.zip' , conf.path);
            unzip('vcapg2w64.zip' , conf.path);
        end
    else
        if(strcmp(C , 'GLNX86'))
            
        elseif(strcmp(C , 'GLNXA64'))
            
        end
    end
end