function parameters = coeff_edit_data_callback2(parameters)

parameters.dictionnary.currentpattern(1,2) = str2double(get(parameters.gui.coeff.editbutton2 , 'String'));