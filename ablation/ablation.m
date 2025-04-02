clear;
clc;
addpath('./fun');
addpath('D:\multiview-dataset');
addpath('./MIndex');
Dataname = 'MSRCv1';
load(Dataname); % 一列一个样本

% 确认数据情况
% Y = truth;
numClust = length(unique(Y));
numSample = length(Y);
numView = length(X);

max_iter = 100;

% 主循环，针对不同的del值进行处理  
for del = [0.1]
    % 确定数据文件路径
    Datafold = [Dataname, '_del_', num2str(del), '.mat'];
    if ~exist(Datafold, 'file')
        % 如果文件不存在，创建文件并写入列标题
        MissIndex(Dataname, numSample, numView, del);
    end
    load(Datafold);

    Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'}; % 性能指标

    % 为不同算法创建结果容器
    all_results = cell(4, 1);
    names = {'PPAU', 'PPAU_P', 'PPAU_E', 'PPAU_A'};
    for alg_idx = 1:4
        all_results{alg_idx} = struct('parameters', [], 'results', []);
    end

    % 针对每个算法分别搜索最佳的lambda和alpha
    best_lambda_alpha_per_alg = zeros(4, 2); % 用于存储每个算法的最佳lambda和alpha
    best_acc_per_alg = zeros(4, 1); % 用于存储每个算法在最佳参数下的最佳准确率

    for alg_idx = 1:4
        for lambda = 10.^(-3:5)
            for alpha = 10.^(-5:3)
                for f = 1
                    fold = folds{f};%样本的索引，每列代表一个视图，每行代表一个样本
                    [X1, ind_0, W, G, St2, n_v] = process_view_data(X, fold, lambda);

                    [Z] = execute_algorithm(X1,Y,f, W,G,ind_0, St2, n_v, numClust, numClust, 3, lambda, alpha, max_iter, numClust, alg_idx);
                    pre_labels = kmeans(Z',numClust, 'Start', 'plus','emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
                    % [Y_bar, ~, ~] = svd(Z', 'econ');
                    % Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
                    % % pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
                    % pre_labels = kmeans(real(Y_bar), numClust, 'Start', 'plus','emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
                    res(f, :) = Clustering8Measure(Y, pre_labels) * 100;
                end
                % 计算性能
                numMetrics = length(Metrics);
                % 初始化存储平均数和标准差的数组
                meanMetrics = zeros(1, numMetrics);
                stdMetrics = zeros(1, numMetrics);

                % 循环遍历每一列，计算平均数和标准差
                for i = 1:numMetrics
                    meanMetrics(i) = mean(res(:,i));
                    stdMetrics(i) = std(res(:,i));           
                end
                % 更新最佳参数
                if meanMetrics(1) > best_acc_per_alg(alg_idx)
                    best_acc_per_alg(alg_idx) = meanMetrics(1);
                    best_lambda_alpha_per_alg(alg_idx, 1) = lambda;
                    best_lambda_alpha_per_alg(alg_idx, 2) = alpha;
                end
            end
        end
    end


    % 使用各自的最佳参数搜索dim和m
    for alg_idx = 1:4
        for k = 3
            dimflag = false;
            for dim = numClust*(1)
                if dim > numSample
                    dimflag = true; % 超出维度最大值退出标记
                    dim = min(dim,numSample);
                end
                mflag = false;
                for m = numClust*(1:2:5)
                    if m > dim
                        break;
                    end
                    for f = 1:10
                        fold = folds{f};%样本的索引，每列代表一个视图，每行代表一个样本
                        [X1, ind_0, W, G, St2, n_v] = process_view_data(X, fold, best_lambda_alpha_per_alg(alg_idx, 1));
                        [Z] = execute_algorithm(X1,Y,f, W, G, ind_0, St2, n_v, dim, m, k, best_lambda_alpha_per_alg(alg_idx, 1), best_lambda_alpha_per_alg(alg_idx, 2), max_iter, numClust, alg_idx);
                        pre_labels = kmeans(Z',numClust, 'Start', 'plus','emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
                        % [Y_bar, ~, ~] = svd(Z', 'econ');
                        % Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
                        % % pre_labels = kmeans(real(Y_bar), numClu st, 'emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
                        % pre_labels = kmeans(real(Y_bar), numClust, 'Start', 'plus','emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
                        res(f, :) = Clustering8Measure(Y, pre_labels) * 100;
                    end
                    % 计算性能
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

                    % 将当前参数和结果添加到对应算法的结果结构中
                    all_results{alg_idx}.parameters = [all_results{alg_idx}.parameters; best_lambda_alpha_per_alg(alg_idx, 1) best_lambda_alpha_per_alg(alg_idx, 2) dim m k];
                    all_results{alg_idx}.results = [all_results{alg_idx}.results; meanMetrics stdMetrics];

                    if dimflag
                        break;
                    end
                end
            end
        end
        [bestacc,bestindex] = max(all_results{alg_idx}.results(:,1));
        bestpara = all_results{alg_idx}.parameters(bestindex,:);
        disp(alg_idx);
        disp(bestacc);
        disp(bestpara);
        save(['./ablation/',Dataname,'_del',num2str(del),'_ab.mat'],"all_results")
    end

end

disp("消融实验完成");

% 定义处理单个视图数据的函数
function [X1,ind_0, W, G, St2, n_v] = process_view_data(X, fold, lambda)
numSample = length(fold);
numView = length(X);

X1 = cell(numView, 1);
ind_0 = cell(numView, 1);
W = cell(numView, 1);
G = cell(numView, 1);
St2 = cell(numView, 1);
n_v = zeros(numView, 1);

for iv = 1:numView
    if size(X{iv},2) ~= numSample
        X{iv} = X{iv}';
    end
    X1{iv}= NormalizeFea(X{iv},0); %标准化每列特征数据
    ind_1 = find(fold(:,iv) == 1);
    ind_0{iv} = find(fold(:,iv) == 0);
    X1{iv}(:,ind_0{iv}) = []; % 删除标记缺失的样本，构造缺失的视图

    %% 构造索引矩阵
    %----填充索引-----%
    n_v(iv)=length(ind_0{iv}); % 记录每个视图缺失样本的数量
    W{iv}=zeros(n_v(iv),numSample);
    for i = 1:length(ind_0{iv})
        j = ind_0{iv}(i); % j 是 fold 矩阵第iv 列中第 i 个值为 0 的线性索引
        W{iv}(i, j) = 1; % 将矩阵 W 的第i 行第 j 列设置为 1
    end
    %-----缺失索引----%
    linshi_W = diag(fold(:,iv));%将该视图的索引转化成对角矩阵
    linshi_W(:,ind_0{iv}) = []; % 从对角阵中删除与删除样本对应的列
    G{iv} = linshi_W; % 存储缺失视角的的索引矩阵 G{iv}

    X1{iv} = X1{iv}*G{iv}'; % 恢复完整大小视图
    %% 构建散点矩阵
    linshi_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
    St2{iv} = mpower(linshi_St,-0.5); % 逆矩阵的平方根
end
end

% 定义执行特定算法并计算性能指标的函数
function [Z] = execute_algorithm(X1,Y,f, W, G, ind_0, St2, n_v, dim, m, k, lambda, alpha, max_iter, numClust, alg_idx)
switch alg_idx
    case 1
        [Z,~] = PIMV_CBG(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,ind_0);
    case 2
        [Z,~] = PPAU_P(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,ind_0);
    case 3
        [Z,~] = PPAU_E(X1,W,G,St2,n_v,dim,m,k,lambda,alpha,max_iter,ind_0);
    case 4
        [Z,~] = PPAU_A(X1,W,G,St2,n_v,dim,m,k,lambda,alpha,max_iter,ind_0);
end

end


% clear;
% clc;
% addpath('./fun');
% addpath('D:\multiview-dataset');
% addpath('./MIndex');
% Dataname='MSRCv1';
% load(Dataname);         % 一列一个样本
%
% % 确认数据情况
% % Y = truth;
% numClust=length(unique(Y));
% numSample=length(Y);
% numView=length(X);
%
% max_iter = 100;
%
% for del = [0.5,0.7]
%     % paired = 1 - del;
%     % Datafold=[Dataname,'_paired_',num2str(del),'.mat'];
%     Datafold=[Dataname,'_del_',num2str(del),'.mat'];
%     if ~exist(Datafold, 'file')
%         % 如果文件不存在，创建文件并写入列标题
%         MissIndex(Dataname,numSample,numView,del);
%     end
%     load(Datafold);
%
%     % 定义列标题
%     columnTitles = {'Lambda', 'Alpha', 'Dimension','m','k','ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy',    'std1','std2','std3','std4','std5','std6','std7','std8'};
%
%     % 为不同算法创建结果容器
%     all_results = cell(4, 1);
%     names = {'PPAU', 'PPAU_P', 'PPAU_E', 'PPAU_A'};
%     for alg_idx = 1:4
%         all_results{alg_idx} = struct('parameters', [], 'results', []);
%     end
%
%     % for k=(1:2:15)
%     for k = 3
%         %% 搜参
%         dimflag = false;
%         for dim = numClust*(1:3)
%         % for dim = numClust*(3)
%             if dim > numSample
%                 dimflag = true; % 超出维度最大值退出标记
%                 dim = min(dim,numSample);
%             end
%             % m=numClust;
%             % m=dim;
%             mflag = false;
%             for m = numClust*(1:3)
%             % for m = numClust*(1)
%             % for m = dim
%                 if m > dim
%                     break;
%                 end
%                 for lambda=10.^(-3:5)
%                 % for lambda=10.^(0)
%                     for alpha=10.^(-5:5)
%                     % for alpha= 10.^(-2)
%                         for f=1
%                             fold = folds{f};%样本的索引，每列代表一个视图，每行代表一个样本
%                             linshi_GG = 0; % 实例存在数
%                             linshi_LS = 0;
%                             for iv = 1:length(X) %循环遍历每个视图
%                                 % X1{iv} = X{iv};
%                                 X1{iv}= NormalizeFea(X{iv}',0); %标准化每列特征数据
%                                 ind_1 = find(fold(:,iv) == 1);
%                                 ind_0 = find(fold(:,iv) == 0);
%                                 X1{iv}(:,ind_0) = []; % 删除标记缺失的样本，构造缺失的视图
%
%                                 %% 构造索引矩阵
%                                 %----填充索引-----%
%                                 n_v(iv)=length(ind_0); % 记录每个视图缺失样本的数量
%                                 W{iv}=zeros(n_v(iv),numSample);
%                                 for i = 1:length(ind_0)
%                                     j = ind_0(i); % j 是 fold 矩阵第 iv 列中第 i 个值为 0 的线性索引
%                                     W{iv}(i, j) = 1; % 将矩阵 W 的第i 行第 j 列设置为 1
%                                 end
%                                 %-----缺失索引----%
%                                 linshi_W = diag(fold(:,iv));%将该视图的索引转化成对角矩阵
%                                 linshi_W(:,ind_0) = []; % 从对角阵中删除与删除样本对应的列
%                                 G{iv} = linshi_W; % 存储缺失视角的索引矩阵 G{iv}
%
%                                 X1{iv} = X1{iv}*G{iv}'; % 恢复完整大小视图
%                                 %% 构建散点矩阵
%                                 linshi_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
%                                 St2{iv} = mpower(linshi_St,-0.5); % 逆矩阵的平方根
%                             end
%
%                             for alg_idx = 1:4
%                                 switch alg_idx
%                                     case 1
%                                         [Z,obj] = PPAU(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);
%                                     case 2
%                                         [Z,obj] = PPAU_P(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);
%                                     case 3
%                                         [Z,obj] = PPAU_E(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);
%                                     case 4
%                                         [Z,obj] = PPAU_A(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);
%                                 end
%                                 pre_labels=kmeans(real(Z'),numClust,'emptyaction','singleton','replicates',10,'display','off');
%                                 % pre_labels=litekmeans(real(Z'),numClust,'MaxIter', 50, 'Replicates', 20);
%                                 % [Y_bar,~,~]=svd(Z','econ');
%                                 % pre_labels=kmeans(real(Y_bar),numClust,'emptyaction','singleton','replicates',20,'display','off');
%                                 % pre_labels=litekmeans(real(Y_bar),numClust,'MaxIter', 50, 'Replicates', 10);
%                                 res(f,:)=Clustering8Measure(Y, pre_labels)*100;
%
%                                 % 计算性能
%                                 Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'}; % 性能指标
%                                 numMetrics = length(Metrics);
%                                 % 初始化存储平均数和标准差的数组
%                                 meanMetrics = zeros(1, numMetrics);
%                                 stdMetrics = zeros(1, numMetrics);
%
%                                 % 循环遍历每一列，计算平均数和标准差
%                                 for i = 1:numMetrics
%                                     meanMetrics(i) = mean(res(:,i));
%                                     stdMetrics(i) = std(res(:,i));
%                                     result{i}=strcat(num2str(meanMetrics(i), '%.2f'),'±',num2str(stdMetrics(i), '%.2f'));
%                                 end
%
%                                 % 将当前参数和结果添加到对应算法的结果结构中
%                                 all_results{alg_idx}.parameters = [all_results{alg_idx}.parameters; lambda alpha dim m k];
%                                 all_results{alg_idx}.results = [all_results{alg_idx}.results; meanMetrics stdMetrics];
%                             end
%                         end
%                     end
%                 end
%             end
%             if dimflag
%                 break;
%             end
%         end
%     end
%     save(['./ablation/',Dataname,'_del',num2str(del),'_ab.mat'],"all_results")
% end
