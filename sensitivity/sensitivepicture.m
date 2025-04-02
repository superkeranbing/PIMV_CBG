% 假设CSV文件名为 'Caltech101_7_del0.5_k3.csv'，并且包含以下列：'Dimension', 'm', 'Lambda', 'Alpha', 'ACC'
filename = 'bbcsport4vbigRnSp_del0.5.csv';

% 读取CSV文件
data = readtable(filename);

% 筛选数据，只保留Dimension等于35且m等于35的行
filteredData = data(data.Dimension == 14 & data.m == 14, :);

% 提取Lambda, Alpha和ACC的值
lambda = filteredData.Lambda;
alpha = filteredData.Alpha;
acc = filteredData.ACC;

% 为了使用bar3，我们需要将数据转换为矩阵形式
% 使用unique函数找到唯一的Lambda和Alpha值，并创建一个索引映射
[uniqueLambda, ~, idxLambda] = unique(lambda, 'stable');
[uniqueAlpha, ~, idxAlpha] = unique(alpha, 'stable');

% 初始化一个矩阵来存储ACC值
accMatrix = NaN(length(uniqueLambda), length(uniqueAlpha));

% 填充accMatrix矩阵
for i = 1:length(acc)
    l = idxLambda(i);
    a = idxAlpha(i);
    accMatrix(l, a) = acc(i);
end

% 绘制3D柱状图
figure;
bar3(accMatrix);
xlabel('\alpha');
ylabel('\lambda');
zlabel('ACC(%)');

% % 设置x轴和y轴的刻度标签为科学计数法格式，并去除末尾的无效零
% xTickLabels = num2str(uniqueAlpha, '%.0e');
% yTickLabels = num2str(uniqueLambda, '%.0e');

% 设置x轴和y轴的刻度标签为科学计数法
xTickLabels = arrayfun(@(x) num2str(x, '%.0e'), uniqueAlpha, 'UniformOutput', false);
yTickLabels = arrayfun(@(y) num2str(y, '%.0e'), uniqueLambda, 'UniformOutput', false);

% 由于arrayfun返回的是单元数组，我们需要将单元数组转换为字符串数组
xTickLabelsStr = cell2mat(xTickLabels);
yTickLabelsStr = cell2mat(yTickLabels);

% 设置刻度标签
set(gca, 'XTickLabels', xTickLabelsStr);
set(gca, 'YTickLabels', yTickLabelsStr);

% 你可能需要调整XTick和YTick来确保标签正确对齐
set(gca, 'XTick', 1:length(uniqueAlpha));
set(gca, 'YTick', 1:length(uniqueLambda));
