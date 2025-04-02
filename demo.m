clear;
% clc;
addpath('./fun');
addpath('D:/multiview-dataset');
addpath('./MIndex');

Dataname='ORL_2';
load(Dataname);         % 一列一个样本
% Y = truth;
numClust=length(unique(Y));
numSample=length(Y);
numView=length(X);

del= 0.1;
max_iter = 100;

Datafold=[Dataname,'_del_',num2str(del),'.mat'];
% Datafold=[Dataname,'_paired_',num2str(1-del),'.mat'];
if ~exist(Datafold, 'file')
    % 如果文件不存在，创建文件并写入列标题
    % MissIndex(Dataname,numSample,numView,del);
end
load(Datafold);

lambda=10^(0);alpha=10^(0);dim=numClust*1;k=3;
m=dim;
% m=numClust*3;
tic;
for f=1:10
    fold = folds{f};%样本的索引，每列代表一个视图，每行代表一个样本
    linshi_GG = 0; % 实例存在数
    linshi_LS = 0;
    ind_0 = {};
    for iv = 1:length(X) %循环遍历每个视图
        %X1{iv} = X{iv};
        X1{iv}= NormalizeFea(X{iv}',0); %标准化数据
        ind_1 = find(fold(:,iv) == 1);
        ind_0{iv} = find(fold(:,iv) == 0);
        X1{iv}(:,ind_0{iv}) = []; % 删除标记缺失的样本，构造缺失的视图
        %% 构造索引矩阵
        %----填充索引-----%
        n_v(iv)=length(ind_0{iv});
        W{iv}=zeros(n_v(iv),numSample);
        for i = 1:length(ind_0{iv})
            j = ind_0{iv}(i); % j 是 fold 矩阵第 iv 列中第 i 个值为 0 的线性索引
            W{iv}(i, j) = 1; % 将矩阵 W 的第i 行第 j 列设置为 1
        end
        %-----缺失索引----%
        linshi_W = diag(fold(:,iv));%将该视图的索引转化成对角矩阵
        linshi_W(:,ind_0{iv}) = []; % 从对角阵中删除与删除样本对应的列
        G{iv} = linshi_W; % 存储缺失视角的索引矩阵 G{iv}
        
        X1{iv} = X1{iv}*G{iv}'; % 恢复完整大小视图
        %% 构建散点矩阵
        linshi_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
        St2{iv} = mpower(linshi_St,-0.5); % 逆矩阵的平方根
    end
   
    %% PIMV_CAG 算法
    [Z,obj] = PIMV_CBG(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,ind_0);
    [Y_bar,~,~]=svd(Z','econ');
    Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1,size(Y_bar,2));
    pre_labels=kmeans(real(Y_bar),numClust,'emptyaction','singleton','replicates',10,'display','off');
    % pre_labels=litekmeans(real(Z'),numClust,'MaxIter', 50, 'Replicates', 10);
    % pre_labels=kmeans(real(Z'),numClust,'emptyaction','singleton','replicates',10,'display','off');
    % pre_labels=litekmeans(real(Y_bar),numClust,'MaxIter', 50, 'Replicates', 10);
    res(f,:)=Clustering8Measure(Y, pre_labels)*100;
end
time = toc/10;
disp(time);

%% 计算性能
Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'}; % 性能指标
numMetrics = length(Metrics);
% 初始化存储平均数和标准差的数组
meanMetrics = zeros(1, numMetrics);
stdMetrics = zeros(1, numMetrics);

% 循环遍历每一列，计算平均数和标准差
for i = 1:numMetrics
    meanMetrics(i) = mean(res(:,i));
    stdMetrics(i) = std(res(:,i));
    result{i}=strcat(num2str(meanMetrics(i), '%.2f'),'±',num2str(stdMetrics(i), '%.2f'));
end
for i=1:3
    fprintf(['---',Metrics{i},'=',result{i},'---']);
end
fprintf('\n');
