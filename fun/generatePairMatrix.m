function M = generateMatrix(n, v, p)
% 生成成对率索引
% p代表缺失率，1-p为成对率

    % 确保p在0到100之间
    if p < 0 || p > 100
        error('Percentage of 0s should be between 0 and 100');
    end
    p_ratio = p / 100; % 将百分比转换为小数比例

    % 初始化矩阵M，所有元素均为1
    M = ones(n, v);

    % 随机选择一些样本
    miss_to_set = round(n * p_ratio);
    miss_indices = randperm(n, miss_to_set); %从n个样本中随机选择不成对样本位置
    half_M=round(miss_to_set/2);
    half_indices=randperm(miss_to_set, half_M);%从缺失样本中随机挑一半
    half = miss_indices(half_indices);
    half_remain = setdiff(miss_indices, half);
    
    M(half,1) = 0;
    M(half_remain,2) = 0;
end

