 

aa = imread('face1.jpg');
pos  = detector_mlhmslbp_spyr(rgb2gray(aa) , model);
        image(aa);
        hold on
        for i=1:size(pos,2)
             if(pos(4 , i) >= min_detect)
               rectangle('Position', [pos(1,i),pos(2,i),pos(3,i),pos(3,i)], 'EdgeColor', [0,1,0], 'linewidth', 2);
               x = pos(1,i); y = pos(2,i); size = pos(3,i);
               cropIm = aa(y:(y+size),x:(x+size),:);
               imwrite(cropIm,'data71.jpg','jpg');
             end
        end
        hold off