clear;
clc;
%  CSV 文件名，并且包含以下列：'Dimension', 'm', 'Lambda', 'Alpha', 'ACC'
filename = 'buaa_del0.5.csv';

% 读取 CSV 文件
data = readtable(filename);

% 筛选数据，只保留 Dimension 等于 35 且 m 等于 35 的行
filteredData = data(data.k == 3, :);

% 提取 Lambda, Alpha 和 ACC 的值
dimension = filteredData.Dimension;
m = filteredData.m;
acc = filteredData.ACC;

% 为了使用 bar3，我们需要将数据转换为矩阵形式
% 使用 unique 函数找到唯一的 Lambda 和 Alpha 值，并创建一个索引映射
[uniqueLambda, ~, idxLambda] = unique(dimension, 'stable');
[uniqueAlpha, ~, idxAlpha] = unique(m, 'stable');

% 初始化一个矩阵来存储 ACC 值
accMatrix = NaN(length(uniqueLambda), length(uniqueAlpha));

% 填充 accMatrix 矩阵
for i = 1:length(acc)
    l = idxLambda(i);
    a = idxAlpha(i);
    accMatrix(l, a) = acc(i);
end

% 删除 X 轴等于 12 的数据
toRemove = find(uniqueAlpha == 12 | uniqueAlpha == 24);
uniqueAlpha(toRemove) = [];
accMatrix(:, toRemove) = [];

% 绘制 3D 柱状图
figure;
bar3(accMatrix);
xlabel('m');
ylabel('dimension');
zlabel('ACC(%)');

% 设置 x 轴和 y 轴的刻度标签为真实值
set(gca, 'XTick', 1:length(uniqueAlpha));
set(gca, 'XTickLabels', num2cell(uniqueAlpha));
set(gca, 'YTick', 1:length(uniqueLambda));
set(gca, 'YTickLabels', num2cell(uniqueLambda));

% 旋转图形，使 X 与 Y 轴调转
view([50 40]);

% ------------------ 保存部分 ------------------
pictuename = ['./picture/',filename,'_dm','.fig'];
% 1. 先保存为 FIG 文件（可编辑）
saveas(gcf, pictuename);  % 或使用 savefig(gcf, 'my_plot.fig');

% % 2. 再用 exportgraphics 输出 PNG（推荐高质量设置）
% exportgraphics(gcf, 'my_plot.png', ...
%     'Resolution', 300, ...      % 分辨率（默认 150 DPI）
%     'BackgroundColor', 'none'   % 透明背景（可选）
% );