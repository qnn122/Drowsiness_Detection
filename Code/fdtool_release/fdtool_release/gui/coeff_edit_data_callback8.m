function parameters = coeff_edit_data_callback8(parameters)

parameters.dictionnary.currentpattern(2,3) = str2double(get(parameters.gui.coeff.editbutton8 , 'String'));