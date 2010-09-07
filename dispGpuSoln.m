function dispGpuSoln()

   in = load('ImageIn.txt'); 
   dilate = load('ImageDilate.txt');  
   erode = load('ImageErode.txt');  
   se = load('asymmPSF_17x17.txt');  

   figure(1); 
   subplot(2,2,1); imagesc(in);  colormap(gray); axis image; title('Original Image');
   subplot(2,2,2); imagesc(se);  colormap(gray); axis image; title('Structuring Element Used');
   subplot(2,2,3); imagesc(dilate);  colormap(gray); axis image; title('Dilated');
   subplot(2,2,4); imagesc(erode); colormap(gray); axis image;  title('Eroded');

   row64_left  = [1:64];
   col64_left  = [1:64];
   row64_right = [1:64];
   col64_right = [193:256];

   figure(2); 
   subplot(3,2,1); imagesc(in(row64_left,  col64_left));
                     colormap(gray); axis image; title('Original Image Left Side');
   subplot(3,2,2); imagesc(in(row64_right, col64_right));
                     colormap(gray); axis image; title('Original Image Right Side');
   subplot(3,2,3); imagesc(dilate(row64_left,  col64_left));
                     colormap(gray); axis image; title('Dilate Left Side');
   subplot(3,2,4); imagesc(dilate(row64_right, col64_right));
                     colormap(gray); axis image; title('Dilate Right Side');
   subplot(3,2,5); imagesc(erode(row64_left,  col64_left));
                     colormap(gray); axis image; title('Erode Left Side');
   subplot(3,2,6); imagesc(erode(row64_right, col64_right));
                     colormap(gray); axis image; title('Erode Image Right Side');


   dilate3x3 = load('Image3x3_dilate.txt');
   erode3x3  = load('Image3x3_erode.txt');
   erodethin  = load('Image3x3_erode_thin.txt');
   figure(3);
   subplot(3,2,1); imagesc(dilate3x3(row64_left,  col64_left));
                     colormap(gray); axis image; title('Dilate3x3 Left Side');
   subplot(3,2,2); imagesc(dilate3x3(row64_right, col64_right));
                     colormap(gray); axis image; title('Dilate3x3 Right Side');
   subplot(3,2,3); imagesc(erode3x3(row64_left,  col64_left));
                     colormap(gray); axis image; title('Erode3x3 Left Side');
   subplot(3,2,4); imagesc(erode3x3(row64_right, col64_right));
                     colormap(gray); axis image; title('Erode3x3 Right Side');
   subplot(3,2,5); imagesc(erodethin(row64_left,  col64_left));
                     colormap(gray); axis image; title('Erode&Thin Left Side');
   subplot(3,2,6); imagesc(erodethin(row64_right, col64_right));
                     colormap(gray); axis image; title('Erode&Thin Right Side');

   
   
   %figure(3);
   %diff = in-gpu;
   %%subplot(1,2,1); imagesc(diff);  colormap(gray); axis image;
   %subplot(1,2,2); imagesc(diff([1:64],[1:64])); colormap(gray); axis image;

