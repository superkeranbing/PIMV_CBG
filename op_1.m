clear;
addpath('./fun');
addpath('D:\multiview-dataset');
addpath('./MIndex');
Dataname = 'buaa';
load(Dataname); % 一列一个样本

% 确认数据情况
Y = truth;
numClust = length(unique(Y));
numSample = length(Y);
numView = length(X);

max_iter = 100;

for del = [0.1]
    Datafold = [Dataname, '_del_', num2str(del), '.mat'];
    if ~exist(Datafold, 'file')
        % 如果文件不存在，创建文件并写入列标题
        % MissIndex(Dataname, numSample, numView, del);
    end
    load(Datafold);

    % 定义列标题
    columnTitles = {'Lambda', 'Alpha', 'Dimension', 'm', 'k',...
        'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'};
    idx = 1;

    for k = (3)
        dimflag = false;
        for dim = numClust*(1)
            if dim > numSample
                dimflag = true;
                dim = min(dim, numSample);
            end

            mflag = false;
            for m = numClust*(1)
                if m > dim
                    break;
                end

                for lambda = 10.^(0)
                    for alpha = 10.^(0)
                        % res = zeros(10, 8);
                        for f = 1
                            fold = folds{f};
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

                            % [Z, obj] = PIMV_CBG(X1, W, St2, n_v_local, dim, m, k, lambda, alpha, max_iter,ind_0);
                            [Z, obj, acc] = PIMV_CBG_c(X1, Y, W, St2, numClust, n_v_local, dim, m, k, lambda, alpha, max_iter,ind_0);
                            [Y_bar, ~, ~] = svd(Z', 'econ');
                            Y_bar = Y_bar ./ repmat(sqrt(sum(Y_bar.^2, 2)), 1, size(Y_bar, 2));
                            pre_labels = kmeans(real(Y_bar), numClust, 'emptyaction', 'singleton', 'replicates', 20);
                            % pre_labels = kmeans(real(Y_bar), numClust, 'replicates', 20);
                            res(f, :) = Clustering8Measure(Y, pre_labels) * 100;
                        end

                        % 计算性能
                        Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'};

                        for i = 1:8
                            result{i} = sprintf('%.2f', res(i));
                        end

                        for i = 1:3
                            fprintf('----%s=%.2f----', Metrics{i}, res(i));
                        end
                        fprintf('\n');

                        % 保存结果
                        hypara = [lambda, alpha, dim, m, k];
                        data(idx, :) = horzcat(hypara, res);
      
                        idx = idx + 1;
                    end
                end
            end

            if dimflag
                break;
            end
        end
    end
    % 创建 Table
    dataTable = array2table(data, 'VariableNames', columnTitles);

    % 保存为 .mat 文件（包含 Table）
    % csvFilePath = ['./sensitivity/', Dataname, '_del', num2str(del), '_k.mat'];
    % save(csvFilePath, 'dataTable');
end

%% 画图
% 假设 obj 和 acc 是你想要绘制的数据向量

figure;

% 首先激活左侧y轴，并在此绘制第一个数据集(obj)
% yyaxis left;
plot(obj(1:50), 'DisplayName', 'obj');
ylabel('Objective function value'); % 设置左侧y轴的标签

% % 然后激活右侧y轴，并在此绘制第二个数据集(acc)
% yyaxis right;
% plot(acc, 'DisplayName', 'acc');
% ylabel('Acc Value'); % 设置右侧y轴的标签

xlabel('Iteration'); % 设置x轴的标签
% legend show; % 显示图例

% % 如果需要设置每个y轴的刻度间隔，可以使用以下命令：
% % 左侧y轴的刻度设置
% yticks('left', 0:步长:最大值);
% % 右侧y轴的刻度设置
% yticks('right', 0:步长:最大值);
