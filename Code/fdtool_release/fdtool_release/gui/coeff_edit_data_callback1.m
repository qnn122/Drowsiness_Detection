function parameters = coeff_edit_data_callback1(parameters)


parameters.dictionnary.currentpattern(1,1) = str2double(get(parameters.gui.coeff.editbutton1 , 'String'));