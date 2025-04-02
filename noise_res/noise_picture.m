clc;
clear;

% 定义 SNR 值
SNR_dB = [-5, 0, 5, 10];

% 定义不同方法的准确率
BSV = [49.33, 49.33, 49.33, 49.33];
Concat = [27.44, 36.00, 29.22, 41.56];
DAIMC = [31.78, 30.78, 42.22, 45.78];
OPTIMC = [27.89, 30.78, 35.33, 37.89];
PIC = [48.44, 50.78, 53.88, 56.56];
SAGF_IMC = [21.67, 21.11, 25.00, 26.11];
PIMVC = [50.56, 52.44, 52.89, 55.44];
SCSL = [37.33, 42.00, 46.78, 52.44];
ours = [62.11, 58.33, 61.44, 59.44];

% 创建 table 变量
noise_performance = table(SNR_dB', BSV', Concat', DAIMC', OPTIMC', PIC', SAGF_IMC', PIMVC', SCSL', ours', ...
    'VariableNames', {'SNR_dB', 'BSV', 'Concat', 'DAIMC', 'OPTIMC', 'PIC', 'SAGF_IMC', 'PIMVC', 'SCSL', 'ours'});

figure; % 创建一个新的图形窗口

% 获取 SNR 列数据
SNR_dB = noise_performance.SNR_dB;

% 遍历表格中的每一列（除了第一列SNR_dB），绘制每种方法的折线图
methods = noise_performance.Properties.VariableNames(2:end); % 获取除SNR_dB外的所有方法名

% 使用 lines 函数生成不同的颜色
colors = lines(length(methods));

hold on; % 启用 hold on 以便在同一个图中绘制多条线
for i = 1:length(methods)
    % 统一使用向上的三角形标记
    plot(SNR_dB, noise_performance.(methods{i}), '-^', ... 
        'Color', colors(i,:), 'MarkerFaceColor', colors(i,:)); % 使用不同的颜色和标记绘制每种方法的折线图
end

% 添加标题和轴标签
title('Performance Comparison at Various SNR Levels');
xlabel('SNR (dB)');
ylabel('Accuracy (%)');

% 添加网格以便更清晰地查看数据点
grid on;

% 添加图例
legend(methods);

% 关闭 hold
hold off;