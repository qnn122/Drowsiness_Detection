function parameters = gui_menu_open_data_callback(parameters)


[parameters.dictionnary.filename , parameters.dictionnary.pathname] =  uigetfile('*.mat', 'Select a dictionnary file');

if (parameters.dictionnary.filename ~= 0)

    temp                                     = fullfile(parameters.dictionnary.pathname, parameters.dictionnary.filename);
    parameters.dictionnary                   = load(temp);
    parameters.dictionnary.filename          = temp;
    
    pattern                                  = unique(parameters.dictionnary.rect_param(1 , :));

    parameters.dictionnary.nP                = length(pattern);

    set([parameters.gui.features.handles' ; parameters.gui.database.handles'] , 'enable' , 'on');


    set(parameters.gui.features.listbutton1 , 'String' , ''); 
    
    set(parameters.gui.features.listbutton1 , 'String' , num2str((1:parameters.dictionnary.nP)') , 'value' , 1); 
    
    parameters                                = display_selected_pattern(parameters);
    
    
    parameters                                = gui_number_of_features(parameters);
    
    set(parameters.gui.database.textbutton4 , 'String' , num2str(parameters.dictionnary.nF));
    
else

    
    
end