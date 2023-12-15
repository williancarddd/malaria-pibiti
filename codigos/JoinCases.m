clc
clear all

basePath = "C:\malaria\Jonathan\Partitions\Dataset01\RGB\backup\";

for i=1:100
    f1 = strcat(basePath, num2str(i), '.csv')
    f2 = strcat(basePath, num2str(i), 'b.csv')
    
    csv1 = readtable(f1);
    
    csv2 = readtable(f2);
    
    toSave = [csv1; csv2];
    writetable(toSave,strcat("C:\malaria\Jonathan\Partitions\Dataset01\RGB\", num2str(i), ".csv"));

end