function parameters = database_edit_callback1(parameters)

parameters.dictionnary.ny                = str2double(get(parameters.gui.database.editbutton1 , 'String'));

parameters                               = gui_number_of_features(parameters);
    
set(parameters.gui.database.textbutton4 , 'String' , num2str(parameters.dictionnary.nF));
