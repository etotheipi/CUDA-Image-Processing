function dispGpuSoln()

   in = load('origImage.txt'); 
   gpu = load('gpu_solution.txt');  

   figure(1); 
   subplot(2,2,1); imagesc(in);  colormap(gray); axis image;
   subplot(2,2,2); imagesc(in([1:64],[1:64]));  colormap(gray); axis image;
   subplot(2,2,3); imagesc(gpu);  colormap(gray); axis image;
   subplot(2,2,4); imagesc(gpu([1:64],[1:64])); colormap(gray); axis image;

   %figure(3);
   %diff = in-gpu;
   %%subplot(1,2,1); imagesc(diff);  colormap(gray); axis image;
   %subplot(1,2,2); imagesc(diff([1:64],[1:64])); colormap(gray); axis image;

