function sF = CameraSetup()
% Purpose:
%   return suitable support format from the camera which is being used
%
% Author: Quang Nguyen
%

info = imaqhwinfo('winvideo');
dev_info = info.DeviceInfo;
suppForm = dev_info.SupportedFormats;

for i = 1:length(suppForm)
    if strcmp(suppForm{i}, 'YUY2_320x240')|strcmp(suppForm{i},'RGB24_320x240')
        sF = suppForm{i};
        break;
    end
end