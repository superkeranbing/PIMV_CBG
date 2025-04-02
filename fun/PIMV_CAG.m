function [Z,obj] = PIMV_CAG(X,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust)
%PPAU 此处显示有关此函数的摘要
% X缺失视图
% W填充索引
% m锚点数
%   此处显示详细说明
rand('seed',6666);

numview=length(X);
numsample = size(X{1},2);

%% 初始化
%-----初始化P------%
for iv = 1:numview
    options = [];
    options.ReducedDim = dim;
    [P1,~] = PCA1(X{iv}', options); % 用PCA得到每个视图初始的投影矩阵
    Piv{iv} = P1'; % 转置后存储
end

%-----初始化A------%
for p = 1 : numview
    A{p} = zeros(dim,m);
end

%-----初始化Z-----%
Z = 0;
XX = [];% 原样本矩阵的concat
for p = 1 : numview
    XX = [XX;X{p}];% 垂直堆叠每个视图的数据矩阵
end
[XU,~,~]=svds(XX',m);

[IDX,~] = litekmeans(XU,m, 'MaxIter',200,'Replicates',30);
for i = 1:numsample
    Z(IDX(i),i) = 1;
end
Z = Z/(m) + (m-1)/m/m; %归一化

%-----初始化E------%
for iv=1:numview
    E{iv}=rand(dim,n_v(iv));
end

%%
flag = 1;
iter = 0;
obj = [];

%%
while flag
    iter = iter + 1;
    
    %% 更新A
    for iv=1:numview
        Y{iv}=Piv{iv}*X{iv}+E{iv}*W{iv};
        B=Y{iv}*Z';
        [u,~,v]=svd(B,'econ');
        A{iv}=u*v';
    end
    %% 更新Z
    ftemp = 0;
    for p = 1 : numview
        ftemp = ftemp - 2 * A{p}' *Y{p};
    end
    
    for j = 1:numsample
        Z_hat = -ftemp(:,j)/2/numview;
        [Z(:,j)] = EProjSimplex_new(Z_hat);
    end
    
    %% 更新P
    for p = 1 : numview
        Y_bar=E{p}*W{p}-A{p}*Z;
        H=-St2{p}*X{p}*Y_bar';
        H(isnan(H)) = 0;
         H(isinf(H)) = 0;
        [linshi_U,~,linshi_V] = svd(H','econ');
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0; 
        Piv{p} = linshi_U*linshi_V'*St2{p};
    end
    
    %% 更新E
    % 缺失样本最近邻G
    for p=1:numview
        linshi_Z=Z*W{p}';
        options=[];
        options.NeighborMode='KNN';
        % options.WeightMode='HeatKernel';
        options.WeightMode='Binary';
        options.k=k;
        % options.k=floor(n_v(iv)/numClust-1);
        G = full(constructW(linshi_Z',options)); % 构建邻接矩阵
        G_graph = (G+G')*0.5;  % 确保对称，对相似性矩阵 W 进行平滑处理（对称化）
        L_g{p}=diag(sum(G_graph))-G_graph;
        C=A{p}*linshi_Z;
        E{p}=C/(eye(n_v(p))+alpha*L_g{p});
    end
    %% -------------- obj --------------- %
    linshi_obj = 0; 
    for iv = 1:numview
        linshi_R=Piv{iv}*X{iv}+E{iv}*W{iv}-A{iv}*Z;
        linshi_obj = linshi_obj+norm(linshi_R,'fro')^2+lambda*norm(Piv{iv},'fro')^2+alpha*trace(E{iv}*L_g{iv}*E{iv}');        
    end
    obj(iter) = linshi_obj; %为什么要与X的范数相除？
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>max_iter || obj(iter) < 1e-10)
        flag = 0;
    end
    
end
end

