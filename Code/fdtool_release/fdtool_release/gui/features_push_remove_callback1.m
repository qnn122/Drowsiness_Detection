function parameters = features_push_remove_callback1(parameters)

selected = get(parameters.gui.features.listbutton1 , 'value');

for i = 1 : length(selected)
    
    index = find(parameters.dictionnary.rect_param(1 , :) == selected(i));
    
    parameters.dictionnary.rect_param(: , index) = [];
    
end

pattern                                  = unique(parameters.dictionnary.rect_param(1 , :));

parameters.dictionnary.nP                = length(pattern);

for i = 1:parameters.dictionnary.nP
  
    index =  find(parameters.dictionnary.rect_param(1 , :) ==  pattern(i) & (pattern(i) ~= i));
    
    parameters.dictionnary.rect_param(1 , index) = i;
    
%    pattern(i) = [];
    
end

set(parameters.gui.features.listbutton1 , 'String' , '' , 'value' , max(1 , selected(1)-1));

set(parameters.gui.features.listbutton1 , 'String' , num2str((1:parameters.dictionnary.nP)') );

if(parameters.dictionnary.nP == 0)
   
    set([parameters.gui.features.listbutton1 , parameters.gui.features.pushbutton2] , 'enable' , 'off');
    
    parameters.dictionnary.currentpattern  = zeros(5,5);
     
end

display_selected_pattern(parameters);

parameters                                = gui_number_of_features(parameters);

set(parameters.gui.database.textbutton4 , 'String' , num2str(parameters.dictionnary.nF));
