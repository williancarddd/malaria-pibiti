clc
clear all

files = dir('C:\malaria\Jonathan\ExperimentoNovo\Metrics\Dataset01\RGB\csvs\*_refined.csv');
csvs{length(files)} = 0;
names{length(files)} = ' ';
mydata(100, length(files)) = 0;
valores{length(10)} = 0;
C(100,4) = 0;
 
for i=1:length(files)
    csvs{i} = readtable(strcat(files(i).folder, '\', files(i).name));
    names{i} = files(i).name;
    valores{i} = csvs{i}.val_f1_score;
    mydata(:, i) = 100*csvs{i}.val_f1_score;
    
    for j=1:100
        C(j, :) = [csvs{i}.val_tp(j), csvs{i}.val_fn(j), ...
            csvs{i}.val_fp(j), csvs{i}.val_tn(j)]/3056;
%         [val1, val2] = calculateMeasures(C, csvs{i}.val_auc(j));
%         val2 - [
%             csvs{i}.val_accuracy(j) csvs{i}.val_precision(j) ...
%             csvs{i}.val_sensitivity(j) csvs{i}.val_fmeasure(j), ...
%             csvs{i}.val_auc(j) csvs{i}.val_specificity(j) ]
    end
    names{i}
    mean(C)
    return
end
% 
% return
% 
% % return
% 
% % Test normality of the data
% [kstest(mydata(:,1)), kstest(mydata(:,2)), kstest(mydata(:,3)), ...
%     kstest(mydata(:,4)), kstest(mydata(:,5)), kstest(mydata(:,6)), ...
%     kstest(mydata(:,7)), kstest(mydata(:,8)) ]
% 
% 
% x = mydata(:,8);
% alpha = 0.05;
% tail = 'right';
% 
% [p1,h1,stats1] = ranksum(x,mydata(:,1), 'tail', tail, 'alpha', alpha);
% [p2,h2,stats2] = ranksum(x,mydata(:,2), 'tail', tail, 'alpha', alpha);
% [p3,h3,stats3] = ranksum(x,mydata(:,3), 'tail', tail, 'alpha', alpha);
% [p3,h3,stats3] = ranksum(x,mydata(:,4), 'tail', tail, 'alpha', alpha);
% 
% 
% [p1, h1;p2, h2; p3,h3]
% 
% return

 
 boxplot(mydata, 'Labels', names);
% return
% hold on
% m = mean(Hausdorf)
% plot(m,1:size(Hausdorf,2), '*','r'); 
ax = gca();                 %axis handle (assumes 1 axis)
bph = ax.Children;          %box plot handle (assumes 1 set of boxplots)
bpchil = bph.Children;      %handles to all boxplot elements
bh = findobj(gca,'type','line','tag','Upper Whisker');
UpperWhisker = reshape([bh.YData], [2,length(files)]);

bh = findobj(gca,'type','line','tag','Lower Whisker');
LowerWhisker = reshape([bh.YData], [2,length(files)]);

h = findobj(gcf,'tag','Median');
medians = cell2mat(get(h,'YData'));
medians = [medians(:,1)'];

% labels = cell2mat(get(h,'Labels'));



h = findobj(gcf,'tag','Outliers');
outliers = get(h,'YData');


tam = size(mydata,1);
size(mydata, 2)
out(size(mydata, 2)) = 0;
trim(size(mydata, 2)) = 0;
m = mean(mydata); % Sample mean
for i=1:size(mydata,2)
    out(i) = size(outliers(i),2)
    trim(i) = mean(rmoutliers(mydata(:,i)));
end
trim

for i=1:length(files)
    length(files)+1-i;
    
    string = string + '%' + names(length(files)-i+1) + newline + " \addplot [black, fill=brazilYellow, boxplot prepared={lower whisker=" + num2str(LowerWhisker(1,i)) + ...
        ",lower quartile=" + num2str(LowerWhisker(2, i)) + ",upper quartile=" + num2str(UpperWhisker(1,i)) + ...
        ",upper whisker=" + num2str(UpperWhisker(2,i)) + ",median=" + num2str(medians(i)) + ",average=" + ...
        sprintf('%2.2f', trim(length(files)-i+1)) + ...
        ",sample size=100""}] coordinates {};" + newline;
end
string