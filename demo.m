clear;
% clc;
addpath('./fun');
addpath('D:/multiview-dataset');
addpath('./MIndex');

Dataname='ORL_2';
load(Dataname);         % Each column represents one sample
% Y = truth; % Ground truth labels
numClust=length(unique(Y));
numSample=length(Y);
numView=length(X);

del= 0.1;
max_iter = 100;

Datafold=[Dataname,'_del_',num2str(del),'.mat'];
% Datafold=[Dataname,'_paired_',num2str(1-del),'.mat'];
if ~exist(Datafold, 'file')
    % If file doesn't exist, create it and write column headers
    % MissIndex(Dataname,numSample,numView,del); % Generate missing index
end
load(Datafold);

lambda=10^(-1);alpha=10^(1);k=3;
% m=dim;
dim=numClust*1;m=numClust*1; % 不能大于最小维度
tic;
for f=1:10
    fold = folds{f}; % Sample indices, each column represents a view, each row represents a sample
    temp_GG = 0; % Instance existence count
    temp_LS = 0;
    ind_0 = {};
    for iv = 1:length(X) % Iterate through each view
        %X1{iv} = X{iv};
        X1{iv}= NormalizeFea(X{iv}',0); % Normalize data
        ind_1 = find(fold(:,iv) == 1); % Available samples
        ind_0{iv} = find(fold(:,iv) == 0); % Missing samples
        X1{iv}(:,ind_0{iv}) = []; % Remove missing samples to construct incomplete view
        %% Construct index matrix
        %----Filling index-----%
        n_v(iv)=length(ind_0{iv}); % Number of missing samples
        W{iv}=zeros(n_v(iv),numSample); % Initialize index matrix
        for i = 1:length(ind_0{iv})
            j = ind_0{iv}(i); % Linear index of i-th zero value in column iv of fold matrix
            W{iv}(i, j) = 1; % Set element (i,j) of matrix W to 1
        end
        %----Missing index----%
        temp_W = diag(fold(:,iv)); % Convert view index to diagonal matrix
        temp_W(:,ind_0{iv}) = []; % Remove columns corresponding to deleted samples
        G{iv} = temp_W; % Store index matrix G{iv} for missing view
        
        X1{iv} = X1{iv}*G{iv}'; % Recover full-size view
        %% Construct scatter matrix
        temp_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
        St2{iv} = mpower(temp_St,-0.5); % Square root of inverse matrix
    end
   
    %% PIMV_CBG algorithm
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

%% Calculate performance metrics
Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'}; % Performance metrics
numMetrics = length(Metrics);
% Initialize arrays to store means and standard deviations
meanMetrics = zeros(1, numMetrics);
stdMetrics = zeros(1, numMetrics);

% Calculate mean and standard deviation for each metric
for i = 1:numMetrics
    meanMetrics(i) = mean(res(:,i));
    stdMetrics(i) = std(res(:,i));
    result{i}=strcat(num2str(meanMetrics(i), '%.2f'),'±',num2str(stdMetrics(i), '%.2f'));
end
for i=1:3
    fprintf(['---',Metrics{i},'=',result{i},'---']);
end
fprintf('\n');
