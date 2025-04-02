function [Z, obj, acc] = PIMV_CBG_c(X, truth, W, St2, numClust, n_v, dim, m, k, lambda, alpha, max_iter,ind_0)

options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'Binary';

numview = length(X);
numsample = size(X{1}, 2);
rand('seed',8888);

% 初始化
Piv = cell(1, numview);
A = cell(1, numview);
E = cell(1, numview);

for iv = 1:numview
    options = [];
    options.ReducedDim = dim;
    [P1, ~] = PCA1(X{iv}', options);
    Piv{iv} = P1';
    A{iv} = zeros(dim, m);
end
%%
% Z = 0;
% XX = [];
% for p = 1:numview
%     XX = [XX; X{p}];
% end
% [XU, ~, ~] = svds(XX', m);
% [IDX, ~] = kmeans(XU, m, 'MaxIter', 200, 'Replicates', 20);
% Z = zeros(m, numsample);
% for i = 1:numsample
%     Z(IDX(i), i) = 1;
% end
% Z = Z / m + (m - 1) / m / m;

Z = 0;
fused_feat = zeros(size(X{1},2),m); % 初始化融合特征矩阵

% 分视图处理
for p = 1:numview
    % 对每个视图单独降维
    [U_p, ~, ~] = svds(X{p}', m); % 执行SVD获取m维特征
    
    % 拼接各视图特征（样本级别融合）
    % fused_feat = [fused_feat, U_p]; % 维度变为 n × (m*numview)
    fused_feat = fused_feat + U_p;
end

% 可选：二次降维到m维（若需要）
% [XU, ~, ~] = svds(fused_feat, m);

% 直接使用拼接特征进行聚类（保持原逻辑）
[IDX, ~] = litekmeans(fused_feat, m, 'MaxIter', 200, 'Replicates', 20);

% 生成初始化矩阵（保持原逻辑）
Z = zeros(m, numsample);
for i = 1:numsample
    Z(IDX(i), i) = 1;
end
Z = Z / m + (m - 1) / m / m; % 平滑处理

% 初始化E
% options.k = 1;
% linshi_G = full(constructW(Z', options));
for iv = 1:numview
    E{iv} = rand(dim, n_v(iv));
    
    % E{iv} = (Piv{iv}*X{iv})*linshi_G*W{iv}';
    % E{iv} = zeros(dim, n_v(iv));
end

flag = 1;
iter = 0;
obj = [];

while flag
    iter = iter + 1;

    % 更新A
    for iv = 1:numview
        Y{iv} = Piv{iv} * X{iv} + E{iv} * W{iv};
        B = Y{iv} * Z';
        [u, ~, v] = svd(B, 'econ');
        A{iv} = u * v';
    end

    % 更新E
    for p = 1:numview
        % linshi_Z = Z * W{p}';
        linshi_Z = Z(:,ind_0{p});
        options.k = k;
        G = full(constructW(linshi_Z', options));
        G_graph = (G + G') * 0.5;
        L_g{p} = diag(sum(G_graph)) - G_graph;
        C = A{p} * linshi_Z;
        E{p} = C / (eye(n_v(p)) + alpha * L_g{p});
    end

    % 更新Z
    ftemp = 0;
    for p = 1:numview
        ftemp = ftemp - 2 * A{p}' * Y{p};
    end

    for j = 1:numsample
        Z_hat = -ftemp(:, j) / (2 * numview);
        Z(:, j) = EProjSimplex_new(Z_hat);
    end

    % 更新P
    for p = 1:numview
        Y_bar = E{p} * W{p} - A{p} * Z;
        H = -St2{p} * (X{p} * Y_bar');
        H(isnan(H)) = 0;
        H(isinf(H)) = 0;
        [linshi_U, ~, linshi_V] = svd(H', 'econ');
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0;
        Piv{p} = (linshi_U * linshi_V') * St2{p};
    end

    [Y_bar, ~, ~] = svd(Z', 'econ');
    Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
    pre_labels = kmeans(real(Y_bar), numClust , 'emptyaction', 'singleton', 'replicates', 20, 'display', 'off');
    metric = Clustering8Measure(truth, pre_labels) * 100;
    acc(iter,:) = metric(1);

    % 计算目标函数
    linshi_obj = 0;
    for iv = 1:numview
        linshi_R = Piv{iv} * X{iv} + E{iv} * W{iv} - A{iv} * Z;
        linshi_obj = linshi_obj + norm(linshi_R, 'fro')^2 + lambda * norm(Piv{iv}, 'fro')^2 + alpha * trace(E{iv} * L_g{iv} * E{iv}');
    end
    obj(iter) = linshi_obj;

    if (iter > 50) && (abs((obj(iter - 1) - obj(iter)) / (obj(iter - 1))) < 1e-6 || iter > max_iter || obj(iter) < 1e-10)
        flag = 0;
    end
end
end