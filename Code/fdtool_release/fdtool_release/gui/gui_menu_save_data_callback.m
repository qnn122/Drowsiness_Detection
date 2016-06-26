function parameters = gui_menu_save_data_callback(parameters)


[parameters.dictionnary.filename , parameters.dictionnary.pathname] =  uiputfile('*.mat', 'Save dictionnary as ');

if (parameters.dictionnary.filename ~= 0)

     tempfile                                                       = fullfile(parameters.dictionnary.pathname, parameters.dictionnary.filename);

     rect_param                                                     = parameters.dictionnary.rect_param;
     
     save(tempfile , 'rect_param');
    
end