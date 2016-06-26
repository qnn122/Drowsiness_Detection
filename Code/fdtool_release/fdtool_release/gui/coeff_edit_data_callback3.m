function parameters = coeff_edit_data_callback3(parameters)

parameters.dictionnary.currentpattern(1,3) = str2double(get(parameters.gui.coeff.editbutton3 , 'String'));