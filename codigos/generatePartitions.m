clc
clear all

DatasetName = 'Dataset01\RGB\';
pathName = 'C:\malaria\Jonathan\';

casos = readtable('C:\malaria\Jonathan\IMAGENS laminas\ROIs\novos.csv');
Exam = casos.Exame;
% Class(size(Exam,1),1) = false;
Class = casos.Class;
% Image = casos.ImagePathName;
Image = casos.Image;
% 
% for i=1:size(Exam,1)
% %        exam = char(casos.ImagePathName(i));
% %    exam = exam(9:end);
%    if (strcmp(char(casos.ObjectsCategory(i)), 'red blood cell'))
%        Class(i) = false;
%        exam = strcat('C:\malaria\Jonathan\', DatasetName, 'Healthy\', num2str(casos.Exame(i)), ...
%            '-', num2str(casos.InstanciaExame(i)),'-', num2str(i), '.bmp');
%    else
%        Class(i) = true;
%        exam = strcat('C:\malaria\Jonathan\', DatasetName, 'Plasmodium\', num2str(casos.Exame(i)),...
%            '-', num2str(casos.InstanciaExame(i)),'-', num2str(i), '.bmp');
%    end
% 
%    
% %    Image{i} = exam;
% end


cvp{100} = 0;

unicos = unique(Exam);


tamanho(size(unicos)) = false;


for k=1:100
    k
    clear Train Test tb filename
    Train(size(Exam,1),1) = false;
    Test(size(Exam,1),1) = false;
    
    if (k == 1)
        cvp{k} = cvpartition(tamanho,'Holdout', 0.2);
    else
        cont = 0;
        while 1
            cvp{k} = repartition(cvp{k-1});
            for kk=1:k-1
                stats = isequal(cvp{kk}.test, cvp{k}.test);
            end
            if (~stats)
                break;
            end
            cont = cont + 1;
        end
    end
    
    
    cvpTrain = cvp{k}.training;
    cvpTest = cvp{k}.test;
    
%     [sum(cvpTrain) sum(cvpTest)]
    
    for j = 1:size(unicos)
        for kk = 1:size(Exam,1)
            if (Exam(kk) == unicos(j) && cvpTrain(j) == 1)
                Train(kk) = true;
            elseif (Exam(kk) == unicos(j) && cvpTest(j) == 1)
                Test(kk) = true;
            end
        end
       
    end
    
    
    tb = table(Image, Class, Train, Test);
    filename = strcat(pathName, 'Partitions\', DatasetName, num2str(k,'%2d'), 'b.csv');
    writetable(tb,filename);
end

