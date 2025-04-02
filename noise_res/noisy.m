% 优化参数两两搜索
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
rng('default');
max_iter = 100;
del = 0.1;
k = 3;
lambda_range = 10.^(-3:3);
alpha_range = 10.^(-5:3);
dim_range = numClust*(1:5);
m_range = numClust*(1:5);
 
% 加载数据文件
Datafold = [Dataname, '_del_', num2str(del), '.mat'];
% Datafold = [Dataname, '_paired_', num2str(1-del), '.mat'];
if ~exist(Datafold, 'file')
    % 如果文件不存在，创建文件并写入列标题
    % MissIndex(Dataname, numSample, numView, del);
end
load(Datafold);

% 设置高斯噪声参数
noise_mean = 0;      % 噪声均值
original_view = X{1}; %选择第一个视图
% 噪声标准差，根据数据尺度调整
target_snr = -5; % 目标SNR（单位：dB）
signal_power = mean(original_view(:).^2);
noise_power = signal_power / (10^(target_snr/10));
noise_std = sqrt(noise_power);

% 选择生成高斯噪声矩阵
noise = noise_mean + noise_std * randn(size(original_view));
X{1} = original_view + noise;
save([Dataname,'_noisy_',num2str(target_snr),'.mat'],'X','truth');
% %% 可视化
% % 每列代表一张10×10的图片
% num_images = 1; % 可视化的图片数量
% image_size = [10, 10]; % 图片尺寸
% 
% % 可视化对比
% figure;
% for i = 1:num_images
%     % 获取第i张原始图片
%     original_image = reshape(original_view(:, i), image_size);
% 
%     % 获取第i张噪声图片
%     noisy_image = reshape(X{2}(:, i), image_size);
% 
%     % 显示原始图片
%     subplot(2, num_images, i);
%     imagesc(original_image);
%     colormap gray; % 使用灰度图
%     axis off;
%     title(['Original ', num2str(i)]);
% 
%     % 显示噪声图片
%     subplot(2, num_images, num_images + i);
%     imagesc(noisy_image);
%     colormap gray; % 使用灰度图
%     axis off;
%     title(['Noisy ', num2str(i)]);
% end
% 
% % 调整布局
% % set(gcf, 'Position', [100, 100, 1200, 600]); % 设置窗口大小

%%
% 标准化文件
X_norm = cell(numView);
for iv = 1:length(X)
    X_norm{iv} = NormalizeFea(X{iv}, 0);
end
clear('X');
 
% 定义列标题
columnTitles = {'Lambda', 'Alpha', 'Dimension', 'm', 'k', 'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', ...
    'Recall', 'AR', 'Entropy', 'std1', 'std2', 'std3', 'std4', 'std5', 'std6', 'std7', 'std8'};
 
%% 第一步：网格搜索 lambda 和 alpha
m_ini = numClust;
dim_ini = numClust;
best_lambda = 0;
best_alpha = 0;
best_acc = -Inf; 
for lambda = lambda_range
    for alpha = alpha_range 
        res = zeros(1, 8);
        for f = 1:10
            % 将 folds 的结果赋值给变量
            % all_folds = folds;
            % fold = all_folds{f};
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
        if meanMetrics(1) > best_acc 
            best_acc = meanMetrics(1);
            best_lambda = lambda;
            best_alpha = alpha;
        end 
    end
end

fprintf('Best lambda: %.2e, Best alpha: %.2e, Best ACC: %.2f\n', best_lambda, best_alpha, best_acc);

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
 
        res = zeros(10, 8);
        for f = 1:10
            % 将 folds 的结果赋值给变量
            % all_folds = folds;
            % fold = all_folds{f};
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
 
%% 保存结果到 .mat 文件
matFilePath = ['./noise_res/', Dataname, '_del', num2str(del), '_k', num2str(k), 'noisy',num2str(target_snr),'.mat'];
save(matFilePath, 'results');
