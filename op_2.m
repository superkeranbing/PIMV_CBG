clear;
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
max_iter = 100;
del = 0.1;
k = 3;
lambda_range = 10.^(-5:3);
alpha_range = 10.^(-5:3);
dim_range = numClust:numClust:numClust*1;
m_range = numClust:numClust:numClust*1;
 
% 加载数据文件
Datafold = [Dataname, '_del_', num2str(del), '.mat'];
if ~exist(Datafold, 'file')
    % 如果文件不存在，创建文件并写入列标题 
    % MissIndex(Dataname, numSample, numView, del);
end
load(Datafold);
 
% 定义列标题
columnTitles = {'Lambda', 'Alpha', 'Dimension', 'm', 'k', 'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', ...
    'Recall', 'AR', 'Entropy', 'std1', 'std2', 'std3', 'std4', 'std5', 'std6', 'std7', 'std8'};
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
 
        for lambda = lambda_range 
            for alpha = alpha_range
                res = zeros(10, 8);
                parfor f = 1:10
                    % 将 folds 的结果赋值给变量
                    all_folds = folds;
                    fold = all_folds{f};
                    X1 = cell(1, numView);
                    W = cell(1, numView);
                    G = cell(1, numView);
                    St2 = cell(1, numView);
                    n_v_local = zeros(1, numView);  % 局部变量 n_v_local
                    ind_0 = {};
 
                    for iv = 1:numView 
                        X1{iv} = NormalizeFea(X{iv}, 0);
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
 
                    [Z, ~] = PIMV_CBG(X1, W, St2, n_v_local, dim, m, k, lambda, alpha, max_iter, ind_0);
                    [Y_bar, ~, ~] = svd(Z', 'econ');
                    Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
                    pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 10, 'display', 'off');
                    res(f, :) = Clustering8Measure(Y, pre_labels) * 100;
                end
 
                % 计算性能
                Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'};
                numMetrics = length(Metrics);
                meanMetrics = mean(res);
                stdMetrics = std(res);
                result = cell(1, numMetrics);
 
                for i = 1:numMetrics
                    result{i} = sprintf('%.2f±%.2f', meanMetrics(i), stdMetrics(i));
                end
 
                for i = 1:3 
                    fprintf('----%s=%s----', Metrics{i}, result{i});
                end
                fprintf('\n');
 
                % 保存结果
                hypara = [lambda, alpha, dim, m, k];
                data = horzcat(hypara, meanMetrics, stdMetrics);
                results = [results; data];
 
                idx = idx + 1;
            end 
        end
    end
end 
 
% 保存结果到 .mat 文件
matFilePath = ['./res/', Dataname, '_del', num2str(del), '_k', num2str(k), '.mat'];
save(matFilePath, 'results', 'columnTitles');