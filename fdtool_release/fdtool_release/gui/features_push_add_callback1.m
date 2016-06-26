function parameters = features_push_add_callback1(parameters)

   
[y , x]      = find(parameters.dictionnary.currentpattern);

if(~isempty(y))
    nR                                                   = length(x);

    h                                                    = max(y) - min(y) + 1;
    w                                                    = max(x) - min(x) + 1;


    parameters.dictionnary.rect_param                    = [parameters.dictionnary.rect_param , zeros(10 , nR)];


    parameters.dictionnary.nP                            = parameters.dictionnary.nP + 1;

    parameters.dictionnary.rect_param(1 , end-nR+1:end)  = parameters.dictionnary.nP;
    parameters.dictionnary.rect_param(2 , end-nR+1:end)  = w;
    parameters.dictionnary.rect_param(3 , end-nR+1:end)  = h;
    parameters.dictionnary.rect_param(4 , end-nR+1:end)  = nR;
    parameters.dictionnary.rect_param(5 , end-nR+1:end)  = (1:nR);
    parameters.dictionnary.rect_param(6 , end-nR+1:end)  = x'-1;
    parameters.dictionnary.rect_param(7 , end-nR+1:end)  = y'-1;
    parameters.dictionnary.rect_param(8 , end-nR+1:end)  = 1;
    parameters.dictionnary.rect_param(9 , end-nR+1:end)  = 1;
    parameters.dictionnary.rect_param(10 , end-nR+1:end) = parameters.dictionnary.currentpattern(find(parameters.dictionnary.currentpattern)');

    parameters                                           = gui_number_of_features(parameters);

    set(parameters.gui.database.textbutton4 , 'String' , num2str(parameters.dictionnary.nF));


    set(parameters.gui.features.listbutton1 , 'String' , num2str((1:parameters.dictionnary.nP)') , 'value' , 1);

    set([parameters.gui.features.listbutton1 , parameters.gui.features.pushbutton2] , 'enable' , 'on');

    display_selected_pattern(parameters);

end
  



