% clear;
% clc;
addpath('./fun');
addpath('D:/multiview-dataset');
addpath('./MIndex');
Dataname='animal';
load(Dataname);         % 一列一个样本

% 确认数据情况
numClust=length(unique(Y));
numSample=length(Y);
numView=length(X);

max_iter = 100;


for del = [0.7]               
    Datafold=[Dataname,'_paired_',num2str(1-del),'.mat'];
    if ~exist(Datafold, 'file')
        % 如果文件不存在，创建文件并写入列标题
        MissIndex(Dataname,numSample,numView,del);
    end
    load(Datafold);

    bestParams = [];
    bestScore = -inf;

    for k=(1)
        %% 搜参
        for dim=numClust*(1)
            m=dim;
            for lambda=10.^(-3:3)
              % for lambda=10.^(1)
                for alpha=10.^(-5:1)
                % for alpha=10.^(-3)
                    % 只使用第一个折叠进行参数搜索
                    idx = randi(1,10);
                    fold = folds{idx}; % 样本的索引，每列代表一个视图，每行代表一个样本
                    [X1, W, G, St2, n_v] = preprocessData(X, fold, lambda);

                    %% PIMV_CAG算法
                    [Z,obj] = PIMV_CAG(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);

                    % 后处理
                    % [Y_bar,~,~] = svd(Z','econ');
                    % Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar,2));
                    % pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 10, 'display', 'off');
                    pre_labels = kmeans(real(Z'), numClust, 'emptyaction', 'singleton', 'replicates', 10, 'display', 'off');
                    
                    % 计算ACC作为评分标准
                    measures = Clustering8Measure(Y, pre_labels);
                    acc = measures(1);  % 假设返回的是一个向量，ACC是第一个元素

                    if acc > bestScore
                        bestScore = acc;
                        bestParams = [lambda, alpha, dim, m, k];
                    end
                end
            end
        end
    end

    % 使用最佳参数运行10个不同的折叠
    lambda = bestParams(1);
    alpha = bestParams(2);
    dim = bestParams(3);
    m = bestParams(4);
    k = bestParams(5);
    results = zeros(length(folds), 8); % 8种度量指标

    for f=1:length(folds)
        fold = folds{f}; % 样本的索引，每列代表一个视图，每行代表一个样本
        [X1, W, G, St2, n_v] = preprocessData(X, fold, lambda);

        %% PIMV_CAG算法
        [Z,obj] = PIMV_CAG(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);

        % 后处理
        % [Y_bar,~,~] = svd(Z','econ');
        % Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar,2));
        % pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 10, 'display', 'off');
        pre_labels = kmeans(real(Z'), numClust, 'emptyaction', 'singleton', 'replicates', 10, 'display', 'off');

        % 计算所有性能指标
        measures = Clustering8Measure(Y, pre_labels);
        results(f, :) = measures * 100;
    end

    % 计算平均值和标准差
    meanResults = mean(results, 1);
    stdResults = std(results, 0, 1);
    
    % % 保存结果到.mat文件
    % saveFileName = ['bestResults_', Dataname, '_paired_', num2str(1-del), '_k', num2str(k), '.mat'];
    % save(saveFileName, 'bestParams', 'meanResults', 'stdResults');
end

function [X1, W, G, St2, n_v] = preprocessData(X, fold, lambda)
    numSample = size(X{1}, 2);
    numView = length(X);
    X1 = cell(1, numView);
    W = cell(1, numView);
    G = cell(1, numView);
    St2 = cell(1, numView);
    n_v = zeros(1, numView);

    for iv = 1:numView % 循环遍历每个视图
        X1{iv} = NormalizeFea(X{iv}, 0); % 标准化数据
        ind_1 = find(fold(:,iv) == 1);
        ind_0 = find(fold(:,iv) == 0);
        X1{iv}(:,ind_0) = []; % 删除标记缺失的样本，构造缺失的视图

        %% 构造索引矩阵
        n_v(iv) = length(ind_0); % 记录每个视图缺失样本的数量
        W{iv} = zeros(n_v(iv), numSample);
        for i = 1:length(ind_0)
            j = ind_0(i); % j 是 fold 矩阵第 iv 列中第 i 个值为 0 的线性索引
            W{iv}(i, j) = 1; % 将矩阵 W 的第i 行第 j 列设置为 1
        end
        linshi_W = diag(fold(:,iv)); % 将该视图的索引转化成对角矩阵
        linshi_W(:,ind_0) = []; % 从对角阵中删除与删除样本对应的列
        G{iv} = linshi_W; % 存储缺失视角的索引矩阵 G{iv}

        X1{iv} = X1{iv} * G{iv}'; % 恢复完整大小视图
        %% 构建散点矩阵
        linshi_St = X1{iv} * X1{iv}' + lambda * eye(size(X1{iv}, 1));
        St2{iv} = mpower(linshi_St, -0.5); % 逆矩阵的平方根
    end
end