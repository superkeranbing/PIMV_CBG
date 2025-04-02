% 优化参数两两搜索
clc;clear;
addpath('./fun');
addpath('D:\multiview-dataset');
addpath('./MIndex');
 
% 数据集名称
Dataname = 'buaa';
load(Dataname); % 一列一个样本
 
% 确认数据情况
Y = truth;
numClust = length(unique(Y));
numSample = length(Y);
numView = length(X);
 
% 参数配置
max_iter = 80;
del = 0.9;
k = 3;
lambda_range = 10.^(-3:5);
alpha_range = 10.^(-5:5);
dim_range = numClust*(1:4);
m_range = numClust*(1:4);
 
% 加载数据文件
Datafold = [Dataname, '_del_', num2str(del), '.mat'];
% Datafold = [Dataname, '_paired_', num2str(1-del), '.mat'];
if ~exist(Datafold, 'file')
    % 如果文件不存在，创建文件并写入列标题
    MissIndex(Dataname, numSample, numView, del);
end
load(Datafold);

% 标准化文件
X_norm = cell(numView);
for iv = 1:length(X)
    X_norm{iv} = NormalizeFea(X{iv}, 0);
end
 
% 定义列标题
columnTitles = {'Lambda', 'Alpha', 'Dimension', 'm', 'k', 'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', ...
    'Recall', 'AR', 'Entropy', 'std1', 'std2', 'std3', 'std4', 'std5', 'std6', 'std7', 'std8'};
 
%% 第一步：网格搜索 lambda 和 alpha
m_ini = numClust;
dim_ini = numClust;
best_lambda = 1e-2;
best_alpha = 1e3;
best_acc = -Inf; 
for lambda = lambda_range
    for alpha = alpha_range 
        res = zeros(1, 8);
        disp(['lambda=',num2str(lambda),'-alpha=',num2str(alpha)]);
        for f = 1:10
            fold = folds{f};
            X1 = cell(1, numView);
            W = cell(1, numView);
            G = cell(1, numView);
            St2 = cell(1, numView);
            n_v_local = zeros(1, numView);  % 局部变量 n_v_local
            ind_0 = {};

            for iv = 1:numView
                X1{iv} = X_norm{iv};
                ind_0{iv} = find(fold(:, iv) == 0);
                X1{iv}(:, ind_0{iv}) = [];

                n_v_local(iv) = length(ind_0{iv});  % 修改局部变量

                W{iv} = zeros(n_v_local(iv), numSample);
                for i = 1:length(ind_0{iv})
                    j = ind_0{iv}(i);
                    W{iv}(i, j) = 1;
                end

                linshi_W = diag(fold(:, iv));
                linshi_W(:, ind_0{iv}) = [];
                G{iv} = linshi_W;

                X1{iv} = X1{iv} * G{iv}';
                linshi_St = X1{iv} * X1{iv}' + lambda * eye(size(X1{iv}, 1));
                St2{iv} = mpower(linshi_St, -0.5);
            end 

            [Z, ~] = PIMV_CBG(X1, W, St2, n_v_local, dim_ini, m_ini, k, lambda, alpha, max_iter, ind_0);
            [Y_bar, ~, ~] = svd(Z', 'econ');
            Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
            % pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
            pre_labels = kmeans(real(Y_bar), numClust, 'Start', 'plus', 'emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
            res(f, :) = Clustering8Measure(Y, pre_labels) * 100;
        end

        % 计算性能
        meanMetrics = mean(res);
        disp(['acc=',num2str(meanMetrics(1))]);
        if meanMetrics(1) > best_acc 
            best_acc = meanMetrics(1);
            best_lambda = lambda;
            best_alpha = alpha;
        end 
    end
end

fprintf('Best lambda: %.2e, Best alpha: %.2e, Best ACC: %.2f\n', best_lambda, best_alpha, best_acc);
currentTimestamp = datetime('now');
disp(currentTimestamp);

%% 第二步：使用最佳 lambda 和 alpha 进行 dim 和 m 的网格搜索
best_dim = 0;
best_m = 0;
best_dim_acc = -Inf;
idx = 1;
results = [];

for dim = dim_range
    if dim > numSample
        dim = min(dim, numSample);
    end
    for m = m_range
        if m > dim
            break;
        end
    disp(['dim=',num2str(dim),'-m=',num2str(m)]);
        res = zeros(10, 8);
        for f = 1:10
            fold = folds{f};
            X1 = cell(1, numView);
            W = cell(1, numView);
            G = cell(1, numView);
            St2 = cell(1, numView);
            n_v_local = zeros(1, numView);  % 局部变量 n_v_local
            ind_0 = {};
 
            for iv = 1:numView
                X1{iv} = X_norm{iv};
                ind_0{iv} = find(fold(:, iv) == 0);
                X1{iv}(:, ind_0{iv}) = [];
 
                n_v_local(iv) = length(ind_0{iv});  % 修改局部变量 
 
                W{iv} = zeros(n_v_local(iv), numSample);
                for i = 1:length(ind_0{iv})
                    j = ind_0{iv}(i);
                    W{iv}(i, j) = 1;
                end
 
                linshi_W = diag(fold(:, iv));
                linshi_W(:, ind_0{iv}) = [];
                G{iv} = linshi_W;
 
                X1{iv} = X1{iv} * G{iv}';
                linshi_St = X1{iv} * X1{iv}' + best_lambda * eye(size(X1{iv}, 1));
                St2{iv} = mpower(linshi_St, -0.5);
            end
 
            [Z, ~] = PIMV_CBG(X1, W, St2, n_v_local, dim, m, k, best_lambda, best_alpha, max_iter, ind_0);
            [Y_bar, ~, ~] = svd(Z', 'econ');
            Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
            % pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
            pre_labels = kmeans(real(Y_bar), numClust, 'Start', 'plus','emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
            res(f, :) = Clustering8Measure(Y, pre_labels) * 100;
        end
 
        % 计算性能
        meanMetrics = mean(res);
        stdMetrics = std(res);
        result = cell(1, 8);
 
        for i = 1:8
            result{i} = sprintf('%.2f±%.2f', meanMetrics(i), stdMetrics(i));
        end
        disp(['acc=',result{1}]);
 
        % 保存结果
        hypara = [best_lambda, best_alpha, dim, m, k];
        data = horzcat(hypara, meanMetrics, stdMetrics);
        results = [results; data];
 
        if meanMetrics(1) > best_dim_acc 
            best_dim_acc = meanMetrics(1);
            best_dim = dim;
            best_m = m;
        end
 
        idx = idx + 1;
    end 
end
 
fprintf('Best dim: %d, Best m: %d, Best ACC: %.2f\n', best_dim, best_m, best_dim_acc);
datetime('now')
 
%% 保存结果到 .mat 文件
matFilePath = ['./res/', Dataname, '_del', num2str(del), '_k', num2str(k), '.mat'];
save(matFilePath, 'results');
