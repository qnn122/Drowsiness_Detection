function parameters = gui_features_dictionary

%
% Create pattern dictionary file with a GUI
%
%  Usage
%  ------
%
%  parameters = gui_features_dictionary;
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009

if ((nargout == 0) || (nargin > 0))
    
    exit('Usage  :  parameters = gui_features_dictionary;');
    
end

warning off

config_gui;



parameters.gui.figure.fig    = figure('visible' , 'off');

set(parameters.gui.figure.fig , 'position', [parameters.gui.figure.offset_fig parameters.gui.figure.size_fig], 'Name' , parameters.gui.figure.title_ihm, 'color', parameters.gui.figure.color_fig , 'Visible', parameters.gui.figure.onoff{parameters.gui.figure.choixvisible} , 'Interruptible', 'off');

set(parameters.gui.figure.fig  , 'HandleVisibility' , 'off' , 'NumberTitle', 'off', 'menubar', 'none', 'renderer', 'opengl', 'doublebuffer', 'on' , 'BackingStore' , 'off' , 'Resize', 'off'  );

set(parameters.gui.figure.fig , 'BusyAction', 'queue' , 'CloseRequestFcn' , 'selection = questdlg(''Quit GUI HAAR ?'' , ''Quit '' , ''Yes'' , ''No'' , ''Yes'' );, switch selection, case ''Yes'' , delete(parameters.gui.figure.fig) , case ''No'', return, end;');



if (isempty(get(parameters.gui.figure.fig , 'children')))
     
    parameters.gui.figure.menu1 = uimenu(parameters.gui.figure.fig , 'Label' , 'Files');
    
    uimenu(parameters.gui.figure.menu1 , 'Label' , 'Open Features Dictionnary' , 'Callback', 'parameters = gui_menu_open_data_callback(parameters);');
    
    uimenu(parameters.gui.figure.menu1 , 'Label' , 'Save Features Dictionnary', 'Callback' , 'parameters = gui_menu_save_data_callback(parameters);');
    
    
end


parameters.gui.frame.leftframe                            = uicontrol(parameters.gui.figure.fig , 'Style', 'Frame', 'Position', parameters.gui.frame.framelposition , 'backgroundcolor' ,parameters.gui.figure.col_framel , 'ForegroundColor' , parameters.gui.figure.col_framel , 'handlevisibility' , 'off');


%%%%%% Edit coefficients button %%%%%%%%


parameters.gui.coeff.editbutton1                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback1 , 'String' , parameters.gui.coeff.editstring1 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips1, 'Position' , parameters.gui.coeff.editposition1 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton2                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback2 , 'String' , parameters.gui.coeff.editstring2 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips2, 'Position' , parameters.gui.coeff.editposition2 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton3                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback3 , 'String' , parameters.gui.coeff.editstring3 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips3, 'Position' , parameters.gui.coeff.editposition3 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton4                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback4 , 'String' , parameters.gui.coeff.editstring4 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips4, 'Position' , parameters.gui.coeff.editposition4 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton5                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback5 , 'String' , parameters.gui.coeff.editstring5 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips5, 'Position' , parameters.gui.coeff.editposition5 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);



parameters.gui.coeff.editbutton6                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback6 , 'String' , parameters.gui.coeff.editstring6 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips6, 'Position' , parameters.gui.coeff.editposition6 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton7                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback7 , 'String' , parameters.gui.coeff.editstring7 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips7, 'Position' , parameters.gui.coeff.editposition7 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton8                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback8 , 'String' , parameters.gui.coeff.editstring8 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips8, 'Position' , parameters.gui.coeff.editposition8 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton9                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback9 , 'String' , parameters.gui.coeff.editstring9 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips9, 'Position' , parameters.gui.coeff.editposition9 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton10                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback10 , 'String' , parameters.gui.coeff.editstring10 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips10, 'Position' , parameters.gui.coeff.editposition10 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);



parameters.gui.coeff.editbutton11                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback11 , 'String' , parameters.gui.coeff.editstring11 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips11, 'Position' , parameters.gui.coeff.editposition11 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton12                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback12 , 'String' , parameters.gui.coeff.editstring12 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips12, 'Position' , parameters.gui.coeff.editposition12 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton13                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback13 , 'String' , parameters.gui.coeff.editstring13 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips13, 'Position' , parameters.gui.coeff.editposition13 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton14                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback14 , 'String' , parameters.gui.coeff.editstring14 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips14, 'Position' , parameters.gui.coeff.editposition14 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton15                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback15 , 'String' , parameters.gui.coeff.editstring15 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips15, 'Position' , parameters.gui.coeff.editposition15 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);



parameters.gui.coeff.editbutton16                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback16 , 'String' , parameters.gui.coeff.editstring16 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips16, 'Position' , parameters.gui.coeff.editposition16 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton17                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback17 , 'String' , parameters.gui.coeff.editstring17 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips17, 'Position' , parameters.gui.coeff.editposition17 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton18                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback18 , 'String' , parameters.gui.coeff.editstring18 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips18, 'Position' , parameters.gui.coeff.editposition18 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton19                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback19 , 'String' , parameters.gui.coeff.editstring19 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips19, 'Position' , parameters.gui.coeff.editposition19 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton20                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback20 , 'String' , parameters.gui.coeff.editstring20 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips20, 'Position' , parameters.gui.coeff.editposition20 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);




parameters.gui.coeff.editbutton21                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback21 , 'String' , parameters.gui.coeff.editstring21 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips21, 'Position' , parameters.gui.coeff.editposition21 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton22                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback22 , 'String' , parameters.gui.coeff.editstring22 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips22, 'Position' , parameters.gui.coeff.editposition22 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton23                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback23 , 'String' , parameters.gui.coeff.editstring23 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips23, 'Position' , parameters.gui.coeff.editposition23 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton24                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback24 , 'String' , parameters.gui.coeff.editstring24 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips24, 'Position' , parameters.gui.coeff.editposition24 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.coeff.editbutton25                         = uicontrol(parameters.gui.figure.fig , 'Style', 'Edit', 'Callback' , parameters.gui.coeff.editcallback25 , 'String' , parameters.gui.coeff.editstring25 ,'visible' , 'off' , 'enable'  , 'off' , 'TooltipString' , parameters.gui.coeff.edittooltips25, 'Position' , parameters.gui.coeff.editposition25 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);


parameters.gui.coeff.pushbutton1                          = uicontrol(parameters.gui.figure.fig , 'Style', 'Pushbutton', 'visible' , 'off' ,  'enable' , 'off' , 'callback' , parameters.gui.coeff.pushcallback1 , 'String' , parameters.gui.coeff.pushstring1  , 'TooltipString' , parameters.gui.coeff.pushtooltips1, 'Position' , parameters.gui.coeff.pushposition1 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname); 


parameters.gui.coeff.handles                              = [parameters.gui.coeff.editbutton1 , parameters.gui.coeff.editbutton2 , parameters.gui.coeff.editbutton3 , parameters.gui.coeff.editbutton4 , parameters.gui.coeff.editbutton5 ; ...
                                                            parameters.gui.coeff.editbutton6 , parameters.gui.coeff.editbutton7 , parameters.gui.coeff.editbutton8 , parameters.gui.coeff.editbutton9 , parameters.gui.coeff.editbutton10 ; ...
                                                            parameters.gui.coeff.editbutton11 , parameters.gui.coeff.editbutton12 , parameters.gui.coeff.editbutton13 , parameters.gui.coeff.editbutton14 , parameters.gui.coeff.editbutton15 ; ...
                                                            parameters.gui.coeff.editbutton16 , parameters.gui.coeff.editbutton17 , parameters.gui.coeff.editbutton18 , parameters.gui.coeff.editbutton19 , parameters.gui.coeff.editbutton20 ; ...
                                                            parameters.gui.coeff.editbutton21 , parameters.gui.coeff.editbutton22 , parameters.gui.coeff.editbutton23 , parameters.gui.coeff.editbutton24 , parameters.gui.coeff.editbutton25 ];

parameters.dictionnary.currentpattern                     = zeros(5,5); 
parameters.dictionnary.nP                                 = 0;
parameters.dictionnary.rect_param                         = [];
                                                        
%%%%%% List pattern buttons %%%%%%%%
                                                        
                                                        
parameters.gui.features.listbutton1                       = uicontrol(parameters.gui.figure.fig , 'Style', 'listbox' , 'visible' , 'off' , 'enable'  , 'off' , 'max' , parameters.gui.features.listmax1 , 'min' , parameters.gui.features.listmin1 , 'value' , parameters.gui.features.listvalue1 ,  'callback', parameters.gui.features.listcallback1 , 'String' , parameters.gui.features.liststring1 , 'enable'  , 'off' , 'TooltipString' , parameters.gui.features.listtooltips1, 'Position' , parameters.gui.features.listposition1 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);


parameters.gui.features.pushbutton1                       = uicontrol(parameters.gui.figure.fig , 'Style', 'Pushbutton', 'visible' , 'off' ,  'enable' , 'off' , 'callback' , parameters.gui.features.pushcallback1 , 'String' , parameters.gui.features.pushstring1  , 'TooltipString' , parameters.gui.features.pushtooltips1, 'Position' , parameters.gui.features.pushposition1 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.features.pushbutton2                       = uicontrol(parameters.gui.figure.fig , 'Style', 'Pushbutton', 'visible' , 'off' ,  'enable' , 'off' , 'callback' , parameters.gui.features.pushcallback2 , 'String' , parameters.gui.features.pushstring2  , 'TooltipString' , parameters.gui.features.pushtooltips2, 'Position' , parameters.gui.features.pushposition2 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);



parameters.gui.features.handles                           = [parameters.gui.features.listbutton1 , parameters.gui.features.pushbutton1 , parameters.gui.features.pushbutton2];                                                   
    

%%%%%% Database buttons %%%%%%%%


parameters.gui.database.textbutton1                       = uicontrol(parameters.gui.figure.fig , 'Style', 'Text' ,  'visible' , 'off' , 'enable'  , 'off', 'String' , parameters.gui.database.textstring1  , 'TooltipString' , parameters.gui.database.texttooltips1, 'Position' , parameters.gui.database.textposition1 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.database.editbutton1                       = uicontrol(parameters.gui.figure.fig , 'Style' , 'Edit' , 'visible' , 'off' , 'enable'  , 'off' , 'Callback' , parameters.gui.database.editcallback1 , 'String' , parameters.gui.database.editstring1  , 'TooltipString' , parameters.gui.database.edittooltips1, 'Position' , parameters.gui.database.editposition1 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);


parameters.gui.database.textbutton2                       = uicontrol(parameters.gui.figure.fig , 'Style', 'Text' ,  'visible' , 'off' , 'enable'  , 'off', 'String' , parameters.gui.database.textstring2  , 'TooltipString' , parameters.gui.database.texttooltips2, 'Position' , parameters.gui.database.textposition2 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.database.editbutton2                       = uicontrol(parameters.gui.figure.fig , 'Style' , 'Edit' , 'visible' , 'off' , 'enable'  , 'off' , 'Callback' , parameters.gui.database.editcallback2 , 'String' , parameters.gui.database.editstring2  , 'TooltipString' , parameters.gui.database.edittooltips2, 'Position' , parameters.gui.database.editposition2 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);


parameters.gui.database.textbutton3                       = uicontrol(parameters.gui.figure.fig , 'Style', 'Text' ,  'visible' , 'off' , 'enable'  , 'off', 'String' , parameters.gui.database.textstring3  , 'TooltipString' , parameters.gui.database.texttooltips3, 'Position' , parameters.gui.database.textposition3 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);

parameters.gui.database.textbutton4                       = uicontrol(parameters.gui.figure.fig , 'Style', 'Text' ,  'visible' , 'off' , 'enable'  , 'off', 'String' , parameters.gui.database.textstring4  , 'TooltipString' , parameters.gui.database.texttooltips4, 'Position' , parameters.gui.database.textposition4 , 'backgroundcolor' , parameters.gui.figure.col_framer , 'fontsize' , parameters.gui.font.fontsize_small , 'fontname' , parameters.gui.font.fontname);


parameters.gui.database.handles                           = [parameters.gui.database.textbutton1 ,parameters.gui.database.editbutton1 , parameters.gui.database.textbutton2 , parameters.gui.database.editbutton2 , parameters.gui.database.textbutton3 , parameters.gui.database.textbutton4];



set(parameters.gui.coeff.handles , 'enable' , 'on' , 'visible' , 'on');

set(parameters.gui.coeff.pushbutton1 , 'enable' , 'on' , 'visible' , 'on');

set(parameters.gui.features.handles , 'visible' , 'on');

set(parameters.gui.features.pushbutton1 , 'enable' , 'on');

set(parameters.gui.database.handles , 'visible' , 'on' , 'enable' , 'on');



warning on
