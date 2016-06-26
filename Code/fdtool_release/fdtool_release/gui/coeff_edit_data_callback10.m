function parameters = coeff_edit_data_callback10(parameters)


parameters.dictionnary.currentpattern(2,5) = str2double(get(parameters.gui.coeff.editbutton10 , 'String'));