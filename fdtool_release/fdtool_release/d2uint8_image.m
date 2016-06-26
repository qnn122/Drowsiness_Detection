function Y = d2uint8_image(I)


minI = min(I(:));
maxI = max(I(:));
Y    = uint8(255*(I - minI)./(maxI-minI));