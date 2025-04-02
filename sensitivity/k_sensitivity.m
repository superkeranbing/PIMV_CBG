clear;

% 假设不同的 del 值为 [0.1, 0.3, 0.5]
dels = [0.1, 0.3, 0.5]; 
Dataname = 'handwritten-5view'; 

figure; 

for del = dels
    csvFilePath = [Dataname,'_del',num2str(del),'_k.mat'];
    load(csvFilePath);
    data = dataTable;
    k = data.k;
    acc = data.ACC;
    hold on;
    plot(k, acc, 'LineStyle', '-', 'Marker', 'o', 'DisplayName', ['del = ', num2str(del)]); 
end

xlabel('Nearest neighbor number'); 
ylabel('ACC(%)'); 

% 设置 y 轴刻度间隔为 10, 如果需要根据实际数据调整，请替换下面的数字
yticks(60:5:100)
legend;