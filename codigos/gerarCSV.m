clc
clear all


myPath = "D:\malaria\Jonathan\";
myFile = "test.json";

myFileName = strcat(myPath, myFile);

fid = fopen(myFileName); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 


dados = jsondecode(str);
tamBase = 86035;

ImageCheckSum(tamBase,1) = "";
ImagePathName(tamBase,1) = "";
ImageShapeR(tamBase,1)   = 0;
ImageShapeC(tamBase,1)   = 0;
ImageShapeChannels(tamBase,1) = 0;
ObjectsCategory(tamBase,1) = "";
ObjectsBoundingBoxMinimumR(tamBase,1) = 0;
ObjectsBoundingBoxMinimumC(tamBase,1) = 0;
ObjectsBoundingBoxMaximumR(tamBase,1) = 0;
ObjectsBoundingBoxMaximumC(tamBase,1) = 0;
InstanciaExame(tamBase,1) = 0;
Exame(tamBase,1) = 0;
Teste(tamBase,1) = false;
Treino(tamBase,1) = false;

contador = 1;
tam1 = size(dados,1);

return

% Teste
for i=1:size(dados,1)
    clear data2
    data2 = dados(i).objects;
   
    
    for j=1:size(data2,1)
        Exame(contador) = i;
        Teste(contador) = true;
        InstanciaExame(contador) = j;
        ImageCheckSum(contador)      = dados(i).image.checksum;
        ImagePathName(contador)      = dados(i).image.pathname;
        ImageShapeR(contador)        = dados(i).image.shape.r;
        ImageShapeC(contador)        = dados(i).image.shape.c;
        ImageShapeChannels(contador) = dados(i).image.shape.channels;
    
        ObjectsCategory(contador)           = data2(j).category;
        ObjectsBoundingBoxMinimumR(contador) = data2(j).bounding_box(1).minimum(1).r;
        ObjectsBoundingBoxMinimumC(contador) = data2(j).bounding_box(1).minimum(1).c;
        ObjectsBoundingBoxMaximumR(contador) = data2(j).bounding_box(1).maximum(1).r;
        ObjectsBoundingBoxMaximumC(contador) = data2(j).bounding_box(1).maximum(1).c;
        contador = contador + 1;
    end
end


myFile = "training.json";

myFileName = strcat(myPath, myFile);

fid = fopen(myFileName); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 

dados = jsondecode(str);

% Treino
for i=1:size(dados,1)
    clear data2
    data2 = dados(i).objects;
    
    for j=1:size(data2,1)
        Exame(contador) = i + tam1;
        Treino(contador) = true;
        InstanciaExame(contador) = j;
        ImageCheckSum(contador)      = dados(i).image.checksum;
        ImagePathName(contador)      = dados(i).image.pathname;
        ImageShapeR(contador)        = dados(i).image.shape.r;
        ImageShapeC(contador)        = dados(i).image.shape.c;
        ImageShapeChannels(contador) = dados(i).image.shape.channels;
    
        ObjectsCategory(contador)            = data2(j).category;
        ObjectsBoundingBoxMinimumR(contador) = data2(j).bounding_box(1).minimum(1).r;
        ObjectsBoundingBoxMinimumC(contador) = data2(j).bounding_box(1).minimum(1).c;
        ObjectsBoundingBoxMaximumR(contador) = data2(j).bounding_box(1).maximum(1).r;
        ObjectsBoundingBoxMaximumC(contador) = data2(j).bounding_box(1).maximum(1).c;
        contador = contador + 1;
    end
end

cases = sortrows(table(Exame,InstanciaExame,Teste,Treino,...
    ImageCheckSum,...
    ImagePathName,...
    ImageShapeR,...
    ImageShapeC,...
    ImageShapeChannels,...
    ObjectsBoundingBoxMinimumR, ...
    ObjectsBoundingBoxMinimumC,...
    ObjectsBoundingBoxMaximumR,...
    ObjectsBoundingBoxMaximumC,...
    ObjectsCategory), 1);

writetable(cases,'casos.csv');
