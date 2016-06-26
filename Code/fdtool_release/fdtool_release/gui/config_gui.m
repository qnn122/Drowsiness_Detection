parameters.gui.font.fontsize                       = 12;
parameters.gui.font.fontsize_small                 = 7;
parameters.gui.font.weight                         = 'bold';
parameters.gui.font.fontname                       = 'MS Sans Serif';


parameters.gui.button.length_button1               = 50;
parameters.gui.button.height_button1               = 20;

parameters.gui.button.length_button2               = 20;
parameters.gui.button.height_button2               = 15;

parameters.plot.handle                             = [];


if(isempty(get(0 , 'CurrentFigure')))
    
    parameters.gui.figure.offset_fig                 = [100 , 60];  % Lunch GUI positions
    
else
    
    temp                                             = get(gcf , 'Position');
    
    parameters.gui.figure.offset_fig                 = temp([1 , 2]);
    
end


parameters.gui.figure.onoff                          = {'off' , 'on'};

parameters.gui.figure.choixvisible                   = 2;

parameters.gui.figure.size_fig                       = [500 200];

parameters.gui.figure.color_fig                      = [211 208 200]/255;

parameters.gui.figure.title_ihm                      = 'Haar Features Designer';

parameters.gui.figure.title_figure                   = 'GUI HAAR';




parameters.gui.figure.col_framel                     = [0.8588235294117647 0.9372549019607843 0.8392156862745098];

parameters.gui.figure.col_frameh                     = [0.8 0.9019607843137255 0.9019607843137255];

parameters.gui.figure.col_frameb                     = [0.7764705882352941 0.8705882352941177 0.9411764705882353];

parameters.gui.figure.col_framer                     = [0.9882352941176471 0.9725490196078431 0.8470588235294118];



% Frame de Gauche %



parameters.gui.frame.decalx_framel                  = 00;            %décalage frame par rapport au bord gauche de l'écran

parameters.gui.frame.decaly_framel                  = 00;            %décalage frame par rapport au bas de l'écran

parameters.gui.frame.length_framel                  = parameters.gui.figure.size_fig(1);           %largeur frame     

parameters.gui.frame.height_framel                  = parameters.gui.figure.size_fig(2); % 590 %hauteur frame taille_fig(2)



parameters.gui.frame.framelposition                 = [parameters.gui.frame.decalx_framel , parameters.gui.frame.decaly_framel , parameters.gui.frame.length_framel , parameters.gui.frame.height_framel ];



parameters.gui.frame.decalx_framer                  = parameters.gui.figure.size_fig(1) - parameters.gui.frame.length_framel;  

parameters.gui.frame.decaly_framer                  = 0; 

parameters.gui.frame.length_framer                  = parameters.gui.frame.length_framel;

parameters.gui.frame.height_framer                  = parameters.gui.figure.size_fig(2); 


parameters.gui.frame.framerposition                 = [parameters.gui.frame.decalx_framer , parameters.gui.frame.decaly_framer , parameters.gui.frame.length_framer , parameters.gui.frame.height_framer ];





parameters.gui.coeff.editstring1                      = '0';

parameters.gui.coeff.edittooltips1                    = 'Enter coefficient';

parameters.gui.coeff.editposition1                    = [parameters.gui.frame.decalx_framel+parameters.gui.frame.length_framel/100, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 1*parameters.gui.button.height_button1 - 15 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback1                    = 'parameters = coeff_edit_data_callback1(parameters);';


parameters.gui.coeff.editstring2                      = '0';

parameters.gui.coeff.edittooltips2                    = 'Enter coefficient';

parameters.gui.coeff.editposition2                    = [parameters.gui.frame.decalx_framel+2*parameters.gui.frame.length_framel/100 + 1*parameters.gui.button.length_button1 , parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 1*parameters.gui.button.height_button1 - 15 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback2                    = 'parameters = coeff_edit_data_callback2(parameters);';


parameters.gui.coeff.editstring3                      = '0';

parameters.gui.coeff.edittooltips3                    = 'Enter coefficient';

parameters.gui.coeff.editposition3                    = [parameters.gui.frame.decalx_framel+3*parameters.gui.frame.length_framel/100 + 2*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 1*parameters.gui.button.height_button1 - 15 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback3                    = 'parameters = coeff_edit_data_callback3(parameters);';


parameters.gui.coeff.editstring4                      = '0';

parameters.gui.coeff.edittooltips4                    = 'Enter coefficient';

parameters.gui.coeff.editposition4                    = [parameters.gui.frame.decalx_framel+4*parameters.gui.frame.length_framel/100 + 3*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 1*parameters.gui.button.height_button1 - 15 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback4                    = 'parameters = coeff_edit_data_callback4(parameters);';


parameters.gui.coeff.editstring5                      = '0';

parameters.gui.coeff.edittooltips5                    = 'Enter coefficient';

parameters.gui.coeff.editposition5                    = [parameters.gui.frame.decalx_framel+5*parameters.gui.frame.length_framel/100 + 4*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 1*parameters.gui.button.height_button1 - 15 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback5                    = 'parameters = coeff_edit_data_callback5(parameters);';





parameters.gui.coeff.editstring6                      = '0';

parameters.gui.coeff.edittooltips6                    = 'Enter coefficient';

parameters.gui.coeff.editposition6                    = [parameters.gui.frame.decalx_framel+parameters.gui.frame.length_framel/100, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 2*parameters.gui.button.height_button1 - 25 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback6                    = 'parameters = coeff_edit_data_callback6(parameters);';


parameters.gui.coeff.editstring7                      = '0';

parameters.gui.coeff.edittooltips7                    = 'Enter coefficient';

parameters.gui.coeff.editposition7                    = [parameters.gui.frame.decalx_framel+2*parameters.gui.frame.length_framel/100 + 1*parameters.gui.button.length_button1 , parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 2*parameters.gui.button.height_button1 - 25 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback7                    = 'parameters = coeff_edit_data_callback7(parameters);';


parameters.gui.coeff.editstring8                      = '0';

parameters.gui.coeff.edittooltips8                    = 'Enter coefficient';

parameters.gui.coeff.editposition8                    = [parameters.gui.frame.decalx_framel+3*parameters.gui.frame.length_framel/100 + 2*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 2*parameters.gui.button.height_button1 - 25 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback8                    = 'parameters = coeff_edit_data_callback8(parameters);';


parameters.gui.coeff.editstring9                      = '0';

parameters.gui.coeff.edittooltips9                    = 'Enter coefficient';

parameters.gui.coeff.editposition9                    = [parameters.gui.frame.decalx_framel+4*parameters.gui.frame.length_framel/100 + 3*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 2*parameters.gui.button.height_button1 - 25 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback9                    = 'parameters = coeff_edit_data_callback9(parameters);';


parameters.gui.coeff.editstring10                      = '0';

parameters.gui.coeff.edittooltips10                    = 'Enter coefficient';

parameters.gui.coeff.editposition10                    = [parameters.gui.frame.decalx_framel+5*parameters.gui.frame.length_framel/100 + 4*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 2*parameters.gui.button.height_button1 - 25 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback10                    = 'parameters = coeff_edit_data_callback10(parameters);';






parameters.gui.coeff.editstring11                      = '0';

parameters.gui.coeff.edittooltips11                    = 'Enter coefficient';

parameters.gui.coeff.editposition11                    = [parameters.gui.frame.decalx_framel+parameters.gui.frame.length_framel/100, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 3*parameters.gui.button.height_button1 - 35 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback11                    = 'parameters = coeff_edit_data_callback11(parameters);';


parameters.gui.coeff.editstring12                      = '0';

parameters.gui.coeff.edittooltips12                    = 'Enter coefficient';

parameters.gui.coeff.editposition12                    = [parameters.gui.frame.decalx_framel+2*parameters.gui.frame.length_framel/100 + 1*parameters.gui.button.length_button1 , parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 3*parameters.gui.button.height_button1 - 35 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback12                    = 'parameters = coeff_edit_data_callback12(parameters);';


parameters.gui.coeff.editstring13                      = '0';

parameters.gui.coeff.edittooltips13                    = 'Enter coefficient';

parameters.gui.coeff.editposition13                    = [parameters.gui.frame.decalx_framel+3*parameters.gui.frame.length_framel/100 + 2*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 3*parameters.gui.button.height_button1 - 35 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback13                    = 'parameters = coeff_edit_data_callback13(parameters);';


parameters.gui.coeff.editstring14                      = '0';

parameters.gui.coeff.edittooltips14                    = 'Enter coefficient';

parameters.gui.coeff.editposition14                    = [parameters.gui.frame.decalx_framel+4*parameters.gui.frame.length_framel/100 + 3*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 3*parameters.gui.button.height_button1 - 35 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback14                    = 'parameters = coeff_edit_data_callback14(parameters);';


parameters.gui.coeff.editstring15                      = '0';

parameters.gui.coeff.edittooltips15                    = 'Enter coefficient';

parameters.gui.coeff.editposition15                    = [parameters.gui.frame.decalx_framel+5*parameters.gui.frame.length_framel/100 + 4*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 3*parameters.gui.button.height_button1 - 35 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback15                    = 'parameters = coeff_edit_data_callback15(parameters);';








parameters.gui.coeff.editstring16                      = '0';

parameters.gui.coeff.edittooltips16                    = 'Enter coefficient';

parameters.gui.coeff.editposition16                    = [parameters.gui.frame.decalx_framel+parameters.gui.frame.length_framel/100, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 4*parameters.gui.button.height_button1 - 45 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback16                    = 'parameters = coeff_edit_data_callback16(parameters);';


parameters.gui.coeff.editstring17                      = '0';

parameters.gui.coeff.edittooltips17                    = 'Enter coefficient';

parameters.gui.coeff.editposition17                    = [parameters.gui.frame.decalx_framel+2*parameters.gui.frame.length_framel/100 + 1*parameters.gui.button.length_button1 , parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 4*parameters.gui.button.height_button1 - 45 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback17                    = 'parameters = coeff_edit_data_callback17(parameters);';


parameters.gui.coeff.editstring18                      = '0';

parameters.gui.coeff.edittooltips18                    = 'Enter coefficient';

parameters.gui.coeff.editposition18                    = [parameters.gui.frame.decalx_framel+3*parameters.gui.frame.length_framel/100 + 2*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 4*parameters.gui.button.height_button1 - 45 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback18                    = 'parameters = coeff_edit_data_callback18(parameters);';


parameters.gui.coeff.editstring19                      = '0';

parameters.gui.coeff.edittooltips19                    = 'Enter coefficient';

parameters.gui.coeff.editposition19                    = [parameters.gui.frame.decalx_framel+4*parameters.gui.frame.length_framel/100 + 3*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 4*parameters.gui.button.height_button1 - 45 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback19                    = 'parameters = coeff_edit_data_callback19(parameters);';


parameters.gui.coeff.editstring20                      = '0';

parameters.gui.coeff.edittooltips20                    = 'Enter coefficient';

parameters.gui.coeff.editposition20                    = [parameters.gui.frame.decalx_framel+5*parameters.gui.frame.length_framel/100 + 4*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 4*parameters.gui.button.height_button1 - 45 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback20                    = 'parameters = coeff_edit_data_callback20(parameters);';






parameters.gui.coeff.editstring21                      = '0';

parameters.gui.coeff.edittooltips21                    = 'Enter coefficient';

parameters.gui.coeff.editposition21                    = [parameters.gui.frame.decalx_framel+parameters.gui.frame.length_framel/100, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 55 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback21                    = 'parameters = coeff_edit_data_callback21(parameters);';


parameters.gui.coeff.editstring22                      = '0';

parameters.gui.coeff.edittooltips22                    = 'Enter coefficient';

parameters.gui.coeff.editposition22                    = [parameters.gui.frame.decalx_framel+2*parameters.gui.frame.length_framel/100 + 1*parameters.gui.button.length_button1 , parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 55 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback22                    = 'parameters = coeff_edit_data_callback22(parameters);';


parameters.gui.coeff.editstring23                      = '0';

parameters.gui.coeff.edittooltips23                    = 'Enter coefficient';

parameters.gui.coeff.editposition23                    = [parameters.gui.frame.decalx_framel+3*parameters.gui.frame.length_framel/100 + 2*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 55 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback23                    = 'parameters = coeff_edit_data_callback23(parameters);';


parameters.gui.coeff.editstring24                      = '0';

parameters.gui.coeff.edittooltips24                    = 'Enter coefficient';

parameters.gui.coeff.editposition24                    = [parameters.gui.frame.decalx_framel+4*parameters.gui.frame.length_framel/100 + 3*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 55 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback24                    = 'parameters = coeff_edit_data_callback24(parameters);';


parameters.gui.coeff.editstring25                      = '0';

parameters.gui.coeff.edittooltips25                    = 'Enter coefficient';

parameters.gui.coeff.editposition25                    = [parameters.gui.frame.decalx_framel+5*parameters.gui.frame.length_framel/100 + 4*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 55 , parameters.gui.button.length_button1, parameters.gui.button.height_button1];

parameters.gui.coeff.editcallback25                    = 'parameters = coeff_edit_data_callback25(parameters);';




parameters.gui.coeff.pushstring1                       = 'Clear pattern';

parameters.gui.coeff.pushtooltips1                     = 'Clear pattern';
 
parameters.gui.coeff.pushposition1                     = [parameters.gui.frame.decalx_framel+2*parameters.gui.frame.length_framel/100+1.8*parameters.gui.button.length_button1, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 6*parameters.gui.button.height_button1 - 75 , 1.8*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1.5];

parameters.gui.coeff.pushcallback1                     = 'parameters = coeff_push_callback1(parameters);';






parameters.gui.features.liststring1                    = {''};

parameters.gui.features.listtooltips1                  = 'Select Features pattern';

parameters.gui.features.listposition1                  = [parameters.gui.frame.decalx_framel+6*parameters.gui.frame.length_framel/100+5*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 7*parameters.gui.button.height_button1 - 15 , 2*parameters.gui.button.length_button1, parameters.gui.button.height_button1*7];


parameters.gui.features.listmax1                       = 2;

parameters.gui.features.listmin1                       = 0;

parameters.gui.features.listvalue1                     = 1;

parameters.gui.features.listcallback1                  = 'parameters = features_list_callback1(parameters);';



parameters.gui.features.pushstring1                       = 'Add Pattern';

parameters.gui.features.pushtooltips1                     = 'Clik to a new pattern ';
 
parameters.gui.features.pushposition1                     = [parameters.gui.frame.decalx_framel+7*parameters.gui.frame.length_framel/100+7*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 2*parameters.gui.button.height_button1 - 15 , 1.8*parameters.gui.button.length_button1, parameters.gui.button.height_button1*2];

parameters.gui.features.pushcallback1                     = 'parameters = features_push_add_callback1(parameters);';




parameters.gui.features.pushstring2                       = 'Remove Pattern(s)';

parameters.gui.features.pushtooltips2                     = 'Clik to a remove pattern(s) ';
 
parameters.gui.features.pushposition2                     = [parameters.gui.frame.decalx_framel+7*parameters.gui.frame.length_framel/100+7*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 4*parameters.gui.button.height_button1 - 20 , 1.8*parameters.gui.button.length_button1, parameters.gui.button.height_button1*2];

parameters.gui.features.pushcallback2                     = 'parameters = features_push_remove_callback1(parameters);';











parameters.gui.database.textstring1                       = 'ny';

parameters.gui.database.texttooltips1                     = 'Enter the number of rows of image database';

parameters.gui.database.textposition1                     = [parameters.gui.frame.decalx_framel+7*parameters.gui.frame.length_framel/100+7*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 30 , 0.7*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1.2];


parameters.gui.database.editstring1                       = '24';

parameters.gui.database.edittooltips1                     = 'Enter the number of rows of image database';

parameters.gui.database.editposition1                     = [parameters.gui.frame.decalx_framel+7*parameters.gui.frame.length_framel/100+7*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 6*parameters.gui.button.height_button1 - 35 , 0.7*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1.2];

parameters.gui.database.editcallback1                     = 'parameters = database_edit_callback1(parameters);';




parameters.gui.database.textstring2                       = 'nx';

parameters.gui.database.texttooltips2                     = 'Enter the number of columns of image database';

parameters.gui.database.textposition2                     = [parameters.gui.frame.decalx_framel+8*parameters.gui.frame.length_framel/100+8*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 5*parameters.gui.button.height_button1 - 30 , 0.7*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1.2];


parameters.gui.database.editstring2                       = '24';

parameters.gui.database.edittooltips2                     = 'Enter the number of columns of image database';

parameters.gui.database.editposition2                     = [parameters.gui.frame.decalx_framel+8*parameters.gui.frame.length_framel/100+8*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 6*parameters.gui.button.height_button1 - 35 , 0.7*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1.2];

parameters.gui.database.editcallback2                     = 'parameters = database_edit_callback2(parameters);';



parameters.gui.database.textstring3                       = 'nF';

parameters.gui.database.texttooltips3                     = 'Number of Total Features';

parameters.gui.database.textposition3                     = [parameters.gui.frame.decalx_framel+6*parameters.gui.frame.length_framel/100+5*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 7*parameters.gui.button.height_button1 - 40 , 2*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1];



parameters.gui.database.textstring4                       = '0';

parameters.gui.database.texttooltips4                     = 'Number of Total Features';

parameters.gui.database.textposition4                     = [parameters.gui.frame.decalx_framel+7*parameters.gui.frame.length_framel/100+7*parameters.gui.button.length_button1 + 10*2, parameters.gui.frame.decaly_framel + parameters.gui.frame.height_framer - 7*parameters.gui.button.height_button1 - 40 , 2*parameters.gui.button.length_button1, parameters.gui.button.height_button1*1];



