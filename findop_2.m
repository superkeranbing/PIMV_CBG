clear;
addpath('./fun');
addpath('D:\multiview-dataset');
addpath('./MIndex');
Dataname='handwritten-5view';
load(Dataname);         % 一列一个样本
% 确认数据情况
% Y = truth;
numClust=length(unique(Y));
numSample=length(Y);
numView=length(X);

max_iter = 100;

for del = [0.1]
    paired = 1 - del;
    % Datafold=[Dataname,'_paired_',num2str(paired),'.mat'];
    Datafold=[Dataname,'_del_',num2str(del),'.mat'];
    if ~exist(Datafold, 'file')
        % 如果文件不存在，创建文件并写入列标题
        MissIndex(Dataname,numSample,numView,del);
    end
    load(Datafold);


    % 定义列标题
    columnTitles = {'Lambda', 'Alpha', 'Dimension','m','k','ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy',    'std1','std2','std3','std4','std5','std6','std7','std8'};
    idx=1;
    % for k=(1:2:15)
    for k = 3
        %% 搜参
        dimflag = false;
        for dim = numClust*(1:4)
            if dim > numSample
                dimflag = true; % 超出维度最大值退出标记
                dim = min(dim,numSample);
            end
            % m=numClust;
            % m=dim;
            mflag = false;
            for m = numClust*(1:4)
            % for m = dim
                if m > dim
                    break;
                end
                for lambda=10.^(0:1)
                    for alpha=10.^(-5:1)
                        for f=1:10
                            fold = folds{f};%样本的索引，每列代表一个视图，每行代表一个样本
                            linshi_GG = 0; % 实例存在数
                            linshi_LS = 0;
                            for iv = 1:length(X) %循环遍历每个视图
                                % X1{iv} = X{iv};
                                X1{iv}= NormalizeFea(X{iv},0); %标准化每列特征数据
                                ind_1 = find(fold(:,iv) == 1);
                                ind_0 = find(fold(:,iv) == 0);
                                X1{iv}(:,ind_0) = []; % 删除标记缺失的样本，构造缺失的视图

                                %% 构造索引矩阵
                                %----填充索引-----%
                                n_v(iv)=length(ind_0); % 记录每个视图缺失样本的数量
                                W{iv}=zeros(n_v(iv),numSample);
                                for i = 1:length(ind_0)
                                    j = ind_0(i); % j 是 fold 矩阵第 iv 列中第 i 个值为 0 的线性索引
                                    W{iv}(i, j) = 1; % 将矩阵 W 的第i 行第 j 列设置为 1
                                end
                                %-----缺失索引----%
                                linshi_W = diag(fold(:,iv));%将该视图的索引转化成对角矩阵
                                linshi_W(:,ind_0) = []; % 从对角阵中删除与删除样本对应的列
                                G{iv} = linshi_W; % 存储缺失视角的索引矩阵 G{iv}

                                X1{iv} = X1{iv}*G{iv}'; % 恢复完整大小视图
                                %% 构建散点矩阵
                                linshi_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
                                St2{iv} = mpower(linshi_St,-0.5); % 逆矩阵的平方根
                            end

                            %% PPAU算法
                            [Z,obj] = PPAU(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);
                            pre_labels=kmeans(real(Z'),numClust,'emptyaction','singleton','replicates',10,'display','off');
                            % pre_labels=litekmeans(real(Z'),numClust,'MaxIter', 50, 'Replicates', 20);
                            % [Y_bar,~,~]=svd(Z','econ');
                            % pre_labels=kmeans(real(Y_bar),numClust,'emptyaction','singleton','replicates',20,'display','off');                         
                            % pre_labels=litekmeans(real(Y_bar),numClust,'MaxIter', 50, 'Replicates', 10);
                            res(f,:)=Clustering8Measure(Y, pre_labels)*100;
                        end

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
                            fprintf(['----',Metrics{i},'=',result{i},'----']);
                        end
                        fprintf('\n');


                        %% 保存结果
                        % 将变量放入一个单元格数组中
                        hypara=[lambda,alpha,dim,m,k];
                        data(idx,:)= horzcat(hypara,meanMetrics,stdMetrics);

                        % CSV文件的路径和名称
                        % csvFilePath = ['./sensitivity/',Dataname,'_del',num2str(del),'_k.csv'];
                        % csvFilePath = ['./res/',Dataname,'_del',num2str(del),'_k',num2str(k),'.csv'];
                        csvFilePath = ['./res/',Dataname,'_del',num2str(del),'_k',num2str(k),'全.csv'];

                        % 检查文件是否已存在
                        if ~exist(csvFilePath, 'file')
                            % 如果文件不存在，创建文件并写入列标题
                            T = array2table(data(1:0,:), 'VariableNames', columnTitles);
                            writetable(T, csvFilePath);
                        end

                        % 将数据和列标题封装成一个表格
                        T = array2table(data(1:idx,:), 'VariableNames', columnTitles);

                        % 写入数据到CSV文件，使用'Append'选项追加数据
                        writetable(T, csvFilePath);

                        idx= idx+1;

                    end
                end
            end
            if dimflag
                break;
            end
        end
    end
end

plot(obj(1:50),'LineStyle', '-', 'Marker', 'o', 'MarkerSize',2); % 设置线条为实线、标记为圆圈并指定大小为 2
xlabel('Iteration');
ylabel('Objective function value');