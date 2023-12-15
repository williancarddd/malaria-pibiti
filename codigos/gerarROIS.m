clc
clear all


myPath = "/media/william/NVME/projects/malaria-pibiti";
myFile = "/media/william/NVME/projects/malaria-pibiti/2_pre_processamento/balanceado.csv";

casos = readtable(myFile);

    pathName = char(casos.ImagePathName(1));
    nomeArquivo = strcat(myPath, 'DatasetOriginal\', pathName(9:end));
%     I = imread(nomeArquivo);
%     imshow(I, []); hold on

tamanhos(size(casos, 1), 4) = 0;
return


for i=1:size(casos, 1)
    
    pos = uint8(casos.Exame(i));
    
    [i]
    
    %     char(casos.ObjectsCategory(i))
        pathName = char(casos.ImagePathName(i));
        nomeArquivo = strcat(myPath, 'DatasetOriginal\', pathName(9:end));
        I = imread(nomeArquivo);
%         imshow(I, []); hold on


            coordenadas = [casos.ObjectsBoundingBoxMinimumC(i), ...
                        casos.ObjectsBoundingBoxMinimumR(i),...
            casos.ObjectsBoundingBoxMaximumC(i),...
                casos.ObjectsBoundingBoxMaximumR(i)];

            rect = [ coordenadas(1),  coordenadas(2), ... 
               coordenadas(3)-coordenadas(1),  coordenadas(4)-coordenadas(2)];
           
%            rectangle('Position',rect,...
%                'EdgeColor', 'red', ...
%          'LineWidth',2,'LineStyle','-'); hold on
%      continue

            rgbCropped = imcrop(I,rect);
            tamanhos(i, 1) = size(rgbCropped,1);
            tamanhos(i, 2) = size(rgbCropped,2);
            Icropped = rgb2gray(rgbCropped);
            Labels(size(rgbCropped,1), size(rgbCropped,2)) = uint8(0);

            borderInc = 2;
            Labels(:,1:borderInc) = 1;
            Labels(1:borderInc,:) = 1;
            Labels(:,end-borderInc:end) = 1;
            Labels(end-borderInc:end,:) = 1;


            x1 = round(size(rgbCropped,2)/2);
            x2 = round(size(rgbCropped,1)/2);
            percentage = 0.8;

            if (x1 > x2)
               inc = round(percentage*x2);
            else
               inc = round(percentage*x1);
            end

       

            Labels(x1-inc:x1+inc, x2-inc:x2+inc) = 2;

            RGB = insertShape(rgbCropped,"circle",[x1 x2 inc],LineWidth==5);
            
         

            t1 = squeeze(RGB(:,:,1)) == 255;
            t2 = squeeze(RGB(:,:,2)) == 255;
            t3 = squeeze(RGB(:,:,3)) == 0;
            t4 = t1&t2&t3;
            imshow(t4==1);
%           
%             return
            
              

%             return
            gray1 = imfill(t4, 'holes');

            box=regionprops(gray1,'BoundingBox');

            rect = [box.BoundingBox];

            rgbCropped = imcrop(rgbCropped,rect);
%             size(rgbCropped)
            
            tamanhos(i, 3) = size(rgbCropped,1);
            tamanhos(i, 4) = size(rgbCropped,2);
            continue
     
 gray1 = imcrop(gray1,rect);
%             imshow(rgbCropped);
%             
% %             imshow(RGB, [])
%             return
% 
%             return
%             Labels(gray1==1) = 2;
            
           
             
          subplot(1,3,1), imshow(rgbCropped, []); title('RGB Image');
            a = rgbCropped(:,:,1);
            a(gray1~=1) = 0;
            rgbCropped(:,:,1) = a;
            b = rgbCropped(:,:,2);
            b(gray1~=1) = 0;
            rgbCropped(:,:,2) = b;
            c = rgbCropped(:,:,3);
            c(gray1~=1) = 0;
            rgbCropped(:,:,3) = c;

%               imshow(rgbCropped);
%               size(rgbCropped)
            continue


            clear filenameSave


        if (strcmp(char(casos.ObjectsCategory(i)), "red blood cell"))
                  filenameSave = strcat(myPath, 'Dataset01\Gray\Healthy\', num2str(casos.Exame(i)), '-', num2str(casos.InstanciaExame(i)),'-', num2str(i), '.bmp');
                 imwrite(rgb2gray(rgbCropped), filenameSave);
    %       rectangle('Position', rect,...
    %       'EdgeColor','b', 'LineWidth', 2); hold on
          
        else
    %         rectangle('Position', [ coordenadas(1),  coordenadas(2), ... 
    %            coordenadas(3)-coordenadas(1),  coordenadas(4)-coordenadas(2)],...
    %       'EdgeColor','r', 'LineWidth', 2); hold on
                filenameSave = strcat(myPath, 'Dataset01\Gray\Plasmodium\', num2str(casos.Exame(i)), '-', num2str(casos.InstanciaExame(i)),'-', num2str(i), '.bmp');
                 imwrite(rgb2gray(rgbCropped), filenameSave);
        end
end
