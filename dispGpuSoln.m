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

   %figure(3);
   %diff = in-gpu;
   %%subplot(1,2,1); imagesc(diff);  colormap(gray); axis image;
   %subplot(1,2,2); imagesc(diff([1:64],[1:64])); colormap(gray); axis image;

