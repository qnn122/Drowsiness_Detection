function parameters = display_selected_pattern(parameters)

selected      = get(parameters.gui.features.listbutton1 , 'value');

index_pattern = find(parameters.dictionnary.rect_param(1 , :) == selected(1));

set(parameters.gui.coeff.handles , 'String' , '0');

parameters.dictionnary.currentpattern = zeros(5,5);

for i = 1:length(index_pattern)
 
    handles_selected = parameters.gui.coeff.handles(parameters.dictionnary.rect_param(7, index_pattern(i))+1:parameters.dictionnary.rect_param(7, index_pattern(i))+parameters.dictionnary.rect_param(9, index_pattern(i))+0,parameters.dictionnary.rect_param(6, index_pattern(i))+1:parameters.dictionnary.rect_param(6, index_pattern(i))+parameters.dictionnary.rect_param(8, index_pattern(i))+0);
    parameters.dictionnary.currentpattern(parameters.dictionnary.rect_param(7, index_pattern(i))+1:parameters.dictionnary.rect_param(7, index_pattern(i))+parameters.dictionnary.rect_param(9, index_pattern(i))+0,parameters.dictionnary.rect_param(6, index_pattern(i))+1:parameters.dictionnary.rect_param(6, index_pattern(i))+parameters.dictionnary.rect_param(8, index_pattern(i))+0) = parameters.dictionnary.rect_param(10, index_pattern(i));
    set(handles_selected , 'String' , num2str(parameters.dictionnary.rect_param(10, index_pattern(i))))

end