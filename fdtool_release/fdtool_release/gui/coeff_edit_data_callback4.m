function parameters = coeff_edit_data_callback4(parameters)

parameters.dictionnary.currentpattern(1,4) = str2double(get(parameters.gui.coeff.editbutton4 , 'String'));