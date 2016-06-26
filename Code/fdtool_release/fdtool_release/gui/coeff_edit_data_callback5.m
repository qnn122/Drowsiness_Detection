function parameters = coeff_edit_data_callback5(parameters)

parameters.dictionnary.currentpattern(1,5) = str2double(get(parameters.gui.coeff.editbutton5 , 'String'));