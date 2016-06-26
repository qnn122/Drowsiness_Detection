function [options , Xfa] = train_stage_cascade(Xtrain , ytrain , Xtest , ytest , options)
%
%
%  Train model for a stage of the cascade
%
%  Usage
%  -----
%
%  [options , Xfa] = train_stage_cascade(Xtrain , ytrain , Xtest , ytest , options)
%
%
%  Inputs
%  ------
%
%
%  Xtrain          Train Data matrix ( Ny x Nx x (options.Npostrain+options.Nnegtrain))
%                  with postives and negatives examples for the current
%                  stage to train
%
%  ytrain          Train labels (1 x (options.Npostrain+options.Nnegtrain))
%
%  Xtest           Trest Data matrix ( Ny x Nx x (options.Npostest+options.Nnegtest))
%                  with postives and negatives examples for testing current
%                  trained stage
%
%  ytest           Test labels (1 x (options.Npostest+options.Nnegtest))
%
%  options         Input options struture (see train_cascade function)
%
%
%  Outputs
%  -------
%
%  options          Updated Options struture (see train_cascade function)
%
%  Xfa              False alarms (non-faces above threshold) (Ny x Nx x Nfa)
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009
%
%  Ref :   [1] M-T. Pham and all, "Detection with multi-exit asymetric boosting", CVPR'08
%  ------  [2] P.A Viola and M. Jones, "Robust real-time face detection",
%              International Journal on Computer Vision, 2004
%
%

Ntrain                       = size(Xtrain , 3);
Ntest                        = size(Xtest , 3);

if(options.cascade_type == 1)
    if(options.typefeat == 0)
        fxtrain              = eval_haar(Xtrain , options);
        if(options.algoboost < 3)
            IItrain          = image_integral_standard(Xtrain);
        else
            Htrain           = haar(Xtrain , options);
        end
    elseif(options.typefeat == 1)
        fxtrain              = eval_mblbp(Xtrain , options);
        Htrain               = mblbp(Xtrain , options);
    end
    
    ctelambda                = -0.5*log(options.lambda);
    
    indptrain                =(ytrain == 1);
    indntrain                = ~indptrain;
    
    wtrain                   = zeros(1 , Ntrain);
    wtrain(indptrain)        = exp(-(fxtrain(indptrain) + ctelambda))/options.Npostrain;
    wtrain(indntrain)        = exp(fxtrain(indntrain)  + ctelambda)/options.Nnegtrain;
    
elseif(options.cascade_type == 0)
    if(options.typefeat == 0)
        IItrain              = image_integral_standard(Xtrain);
    elseif(options.typefeat == 1)
        Htrain               = mblbp(Xtrain , options);
    end
    wtrain                   = ones(1,Ntrain);
end

indptest                      = (ytest==1);
indntest                      = ~indptest;
if(Ntest > 0)
    indp                      = indptest;
    indn                      = indntest;
    Nneg                      = options.Nnegtest;
    Npos                      = options.Npostest;
else
    indp                      = indptrain;
    indn                      = indntrain;
    Nneg                      = options.Nnegtrain;
    Npos                      = options.Npostrain;
end

alpham                        = 1;
betam                         = 1;
alphaold                      = 1;
betaold                       = 1;
K                             = 0;
m                             = 0;

cascade_old                   = options.cascade;
current_stage                 = length(options.m);
alpha0                        = options.alpha0(current_stage);
beta0                         = options.beta0(current_stage);

while( (m < options.maxwl_perstage) && ~((alpham < alpha0) && (betam < beta0)) )
    m                            = m + 1;
    wtrain                       = wtrain/sum(wtrain);
    
    t1                           = tic;
    if(options.typefeat == 0)
        if (options.algoboost == 0)
            [hm , wnew]          = haar_gentle_weaklearner(IItrain , ytrain , wtrain , options);
        elseif (options.algoboost == 1)
            [hm , wnew]          = haar_ada_weaklearner(IItrain , ytrain , wtrain , options);
        elseif (options.algoboost == 2)
            [hm , wnew]          = fast_haar_ada_weaklearner(IItrain , ytrain , wtrain , options);
        elseif (options.algoboost == 3)
            [hm , wnew]          = haar_gentle_weaklearner_memory(Htrain , ytrain , wtrain , options);
        else
            [hm , wnew]          = haar_ada_weaklearner_memory(Htrain , ytrain , wtrain , options);
        end
    elseif(options.typefeat == 1)
        if (options.algoboost == 0)
            [hm , wnew]          = mblbp_gentle_weaklearner(Htrain , ytrain , wtrain , options);
        elseif(options.algoboost == 1)
            [hm , wnew]          = mblbp_ada_weaklearner(Htrain , ytrain , wtrain , options);
        end
    end
    t2                           = toc(t1);
    fprintf('Train weaklearner %d in %6.3f s\n' , m , t2);
    drawnow
    
    options.param                = [options.param , hm];
    options.cascade              = [cascade_old , [m ; 0]];
    
    if(Ntest > 0)
        if(options.typefeat == 0)
            fx                   = eval_haar(Xtest , options);
        elseif(options.typefeat == 1)
            fx                   = eval_mblbp(Xtest , options);
        end
    else
        if(options.typefeat == 0)
            fx                   = eval_haar(Xtrain , options);
        elseif(options.typefeat == 1)
            fx                   = eval_mblbp(Xtrain , options);
        end
    end
    
    if(options.cascade_type == 0)
        ufx                      = unique(fx);
        lufx                     = length(ufx);
        co                       = 1;
        currentthresh            = ufx(co);
        indfa                    = (fx(indn) >= currentthresh);
        if(~isempty(indfa))
            options.fa           = sum(indfa);
        else
            options.fa           = 0.0;
        end
        currentalpha             = options.fa/Nneg;  %False acceptance rate
        indfr                    = (fx(indp) < currentthresh);
        if(~isempty(indfr))
            options.fr           = sum(indfr);
        else
            options.fr           = 0.0;
        end
        currentbeta              = options.fr/Npos;  %False rejection rate
        threshold                = currentthresh;
        alpham                   = currentalpha;
        betam                    = currentbeta;
        
        while((currentbeta < beta0) && (lufx <= (co+1)))
            co                   = co+1;
            threshold            = currentthresh;
            currentthresh        = ufx(co);
            alpham               = currentalpha;
            betam                = currentbeta;
            indfa                = (fx(indn) >= currentthresh);
            options.fa           = sum(indfa);
            currentalpha         = options.fa/Nneg;  %False acceptance rate
            options.fr           = sum(fx(indp) < currentthresh);
            currentbeta          = options.fr/Npos;  %False rejection rate
        end
    elseif(options.cascade_type == 1)
        threshold                = 0.0;
        indfa                    = (fx(indn) >= threshold);
        if(~isempty(indfa))
            options.fa           = sum(indfa);
        else
            options.fa           = 0;
        end
        alpham                   = options.fa/Nneg;  %False acceptance rate
        indfr                    = (fx(indp) < threshold);
        if(~isempty(indfr))
            options.fr           = sum(indfr);
        else
            options.fr           = 0;
        end
        betam                    = options.fr/Npos;  %False rejection rate
    end
    
    if( (alpham == alphaold) || (betam == betaold))
        K                        = K+1;
    else
        alphaold                 = alpham;
        betaold                  = betam;
        K                        = 0;
    end
    
    if(K >= options.maxK)
        break;
    end
    
    wtrain                      = wnew;
    options.indexF(hm(1))       = -1;
    
    fprintf('stage %d/%d, m = %d, alpham = %5.4f\n' , current_stage , options.maxstage , m ,  alpham);
    fprintf('stage %d/%d, m = %d, betam = %5.4f\n'  , current_stage , options.maxstage , m ,  betam);
    drawnow
end


if(options.cascade_type == 0)
    Xfa                                  = Xtest(: , : , indfa);
elseif(options.cascade_type == 1)
    Xfa                                  = Xtrain(: , : , indfa);
end

if(K < options.maxK)
    options.m                            = [options.m , m];
    options.thresholdperstage            = [options.thresholdperstage , threshold];
    options.alphaperstage                = [options.alphaperstage , alpham];
    options.betaperstage                 = [options.betaperstage  , betam];
    options.cascade(2,end)               = threshold;
else
    options.param                        = options.param(: , end-m+1);
end


