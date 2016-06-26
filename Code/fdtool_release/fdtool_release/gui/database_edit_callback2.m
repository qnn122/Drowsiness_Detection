function parameters = database_edit_callback2(parameters)

    
parameters.dictionnary.nx                = str2double(get(parameters.gui.database.editbutton2 , 'String'));

parameters                               = gui_number_of_features(parameters);

set(parameters.gui.database.textbutton4 , 'String' , num2str(parameters.dictionnary.nF));
