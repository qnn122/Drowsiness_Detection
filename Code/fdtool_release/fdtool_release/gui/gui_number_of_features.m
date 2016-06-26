
function parameters                      = gui_number_of_features(parameters)

pattern                                  = unique(parameters.dictionnary.rect_param(1 , :));

parameters.dictionnary.ny                = str2double(get(parameters.gui.database.editbutton1 , 'String'));

parameters.dictionnary.nx                = str2double(get(parameters.gui.database.editbutton2 , 'String'));

h                                        = [];

w                                        = [];

for i = 1:length(pattern) %parameters.dictionnary.nP

    index                                = find(parameters.dictionnary.rect_param(1 , :) == pattern(i));

    w                                    = [w , parameters.dictionnary.rect_param(2 , index(1))];

    h                                    = [h , parameters.dictionnary.rect_param(3 , index(1))];

end

parameters.dictionnary.nF                = nbfeat_haar(parameters.dictionnary.ny , parameters.dictionnary.nx , h , w);
