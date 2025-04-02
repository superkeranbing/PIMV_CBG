addpath("ablation");
clear;
% % 读取 Excel 文件
% data = xlsread('Caltech7消融实验.xlsx');
% 
% % 提取缺失率数据
% missing_rates = data(1, 1:end);
% 
% % 提取不同方法的数据
% no_projection = data(2, 1:end);
% no_imputation = data(3, 1:end);
% no_consensus = data(4, 1:end);
% pimv_cag = data(5, 1:end);
% 
% % 绘制折线图
% figure;
% plot(missing_rates, no_projection, 'LineStyle', '-', 'Marker', 'o', 'DisplayName', 'Without Projection');
% hold on;
% plot(missing_rates, no_imputation, 'LineStyle', '-', 'Marker', 'o', 'DisplayName', 'Without Imputation');
% plot(missing_rates, no_consensus, 'LineStyle', '-', 'Marker', 'o', 'DisplayName', 'Without Anchor Graph');
% plot(missing_rates, pimv_cag, 'LineStyle', '-', 'Marker', 'o', 'DisplayName', 'PIMV\_CAG');
% 
% xlabel('Missing Rate');
% ylabel('ACC(%)');
% legend;

% 读取Excel文件
data = xlsread('MSRCv1消融实验.xlsx');

% 提取缺失率数据
missing_rates = data(1, 1:end);

% 提取不同方法的数据
no_projection = data(2, 1:end);
no_imputation = data(3, 1:end);
no_consensus = data(4, 1:end);
pimv_cag = data(5, 1:end);

% 设置柱子宽度
bar_width = 0.2;

% 计算柱子位置
x_positions = (1:length(missing_rates)) - bar_width*(length([no_projection no_imputation no_consensus pimv_cag])-1)/2;

% 绘制柱状图
figure;
bar(x_positions+(0)*bar_width, no_projection, bar_width, 'DisplayName', 'Without Projection');
hold on;
bar(x_positions+(1)*bar_width, no_imputation, bar_width, 'DisplayName', 'Without Imputation');
bar(x_positions+(2)*bar_width, no_consensus, bar_width, 'DisplayName', 'Without Anchor Graph');
bar(x_positions+(3)*bar_width, pimv_cag, bar_width, 'DisplayName', 'PIMV\_CAG');

% 设置 x 轴刻度标签在四个柱子中间
new_tick_positions = x_positions+0.3;
set(gca,'XTick',new_tick_positions);
set(gca,'XTickLabel',missing_rates);

xlabel('Missing Rate');
ylabel('ACC(%)');
% title('不同方法在不同缺失率下的准确率');
legend('Location', 'best');