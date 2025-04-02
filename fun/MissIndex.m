function[]=MissIndex(dataname,numSample,numView,p)
% 生成缺失索引
% dataname文件名
% numSample样本数
% numView视图数
% p缺失率

p=p*100;
for i=1:10
    folds{i}=generateMatrix(numSample,numView,p); %缺失
    % folds{i}=generatePairMatrix(numSample,numView,p); %成对率1-p%
end
% 定义文件名
filename = ['.\MIndex\',dataname,'_del_',num2str(p/100),'.mat'];
% filename = ['.\MIndex\',dataname,'_paired_',num2str(1-p/100),'.mat'];
% 保存矩阵
save(filename, 'folds');
load(filename);
end