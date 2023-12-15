clc
clear all

contagem(120,2) = 0;


myPath = "D:\malaria\Jonathan\";
myFile = "test.json";
myFile2 = "training.json";

myFileName = strcat(myPath, myFile);
myFileName2 = strcat(myPath, myFile2);

% casos = readtable(myFile);

fid = fopen(myFileName); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 

fid2 = fopen(myFileName2); 
raw2 = fread(fid2,inf); 
str2 = char(raw2'); 
fclose(fid2); 

dados = jsondecode(str);

dados = [dados; jsondecode(str2)];



PosNeg(size(dados,1),3) = 0;

for i=1:size(dados,1)
    clear data2
    data2 = dados(i).objects;
    
    PosNeg(i,3) = size(data2,1);
    
    for j=1:size(data2,1)
        if (strcmp( char(data2(j).category), "red blood cell")) 
            PosNeg(i,1) = PosNeg(i,1) + 1;
        else
            PosNeg(i,2) = PosNeg(i,2) + 1;
        end
        
    end
end