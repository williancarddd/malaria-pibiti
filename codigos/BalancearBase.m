clc
clear all


myPath = "D:\malaria\Jonathan\";
myFile = "casos.csv";

casos = readtable( myFile );

pathName = char( casos.ImagePathName(1) );
nomeArquivo = strcat(myPath, 'Dataset\', pathName(9:end));

load('PosNeg.mat');

PosNegContagem( size( PosNeg, 1), 2) = 0;

infectados = casos(casos.ObjectsCategory ~= "red blood cell", :); 

unicos = unique(infectados.Exame);
selecionados = 0;

for i=1:size(unicos,1)
    i
    clear examPos numel selectedPos selectedExams
    examPos = casos.Exame == unicos(i);
    numel = sum(casos.ObjectsCategory ~= "red blood cell" & casos.Exame == unicos(i)); 
    selectedPos = casos.ObjectsCategory == "red blood cell" & examPos;
    
    selectedExams = casos(selectedPos, :); 
    selectedExams = selectedExams(1:numel, :);
    infectados = [infectados; selectedExams];
end

writetable(infectados,'balanceado.csv');
