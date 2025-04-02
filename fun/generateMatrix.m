function M = generateMatrix(n, v, p)

    % rand('seed',6666);

    % 确保p在0到100之间
    if p < 0 || p > 100
        error('Percentage of 0s should be between 0 and 100');
    end
    p_ratio = p / 100; % 将百分比转换为小数比例

    % 初始化矩阵M，所有元素均为1
    M = ones(n, v);

    % 随机选择一部分元素改为0
    total_elements = n * v;
    num_zeros_to_set = round(total_elements * p_ratio);
    zero_indices = randperm(total_elements, num_zeros_to_set);
    M(zero_indices) = 0;

    % 确保每行至少有一个1
    for i = 1:n
        if sum(M(i, :)) == 0
            % 如果发现某一行全是0，随机选择一个位置改为1
            random_col = randi(v);
            M(i, random_col) = 1;
        end
    end
end

