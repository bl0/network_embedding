% 1 ---- 0.21910 ---- KNN
% 2 ---- 0.58426 ---- linear SVM
% 3 ---- 0.61797 ---- polynomial SVM one-to-one
% 4 ---- 0.61797 ---- PCA & polynomial SVM
% 5 ---- 0.38764 ---- newff 10 0.00001
% 6 ---- 0.50561 ---- newff [10,7,3] 0.000001
% 7 ---- 0.55617 ---- newff learnGDM 10 10e-11 nntool
% 8 ---- 0.61797 ---- polynomial SVM one-to-all
% 9 ---- 0.62921 ---- polynomial SVM one-to-all + 全排列投票
% 10---- 0.67977 ---- polynomial SVM one-to-all + 全排列投票 + 邻接矩阵对角线元素置1
clear;clc;
load('data/airport_edgelist');
airport_sample=importdata('data/airport_sample');
load('data/airport_train');
load('data/airport_test');

train_map = zeros(length(airport_train(:,1))+length(airport_test(:,1)));
for i = 1:length(airport_edgelist(:,2))
    train_map(airport_edgelist(i,1),airport_edgelist(i,2)) = 1;
    train_map(airport_edgelist(i,2),airport_edgelist(i,1)) = 1;
end
train_map = train_map + eye(length(train_map(:,1)));

% mu = mean(train_map);
% Std = std(train_map);
% train_map = train_map - ones(length(train_map(:,1)),1) * mu;
% train_map = train_map./(ones(length(train_map(:,1)),1) * Std + 0.0001);

index = 1;
count = 1;
for i = 1:length(train_map(:,1))
    if find(airport_train(:,1)==i)
        trainset(index,:) = train_map(i,:);
        index = index + 1;
    end
    if find(airport_test(:,1)==i)
        testset(count,:) = train_map(i,:);
        count = count + 1;
    end
end

% % 数据归一化处理
% mu = mean(trainset);
% Std = std(trainset);
% trainset = trainset - ones(length(trainset(:,1)),1) * mu;
% trainset = trainset./(ones(length(trainset(:,1)),1) * Std + 0.0001);
% mu = mean(testset);
% Std = std(testset);
% testset = testset - ones(length(testset(:,1)),1) * mu;
% testset = testset./(ones(length(testset(:,1)),1) * Std + 0.0001);

N = 4;
COM = perms(1:N);

% %随机选出全排列中的部分排列去做投票决策
% VoteNUM = 11;
% temp = randperm(length(COM(:,1)));
% COM = COM(temp(1:VoteNUM),:);

label_test = cell(length(COM(:,1)),1);
for i = 1:length(COM(:,1))
    temp = zeros(length(testset(:,1)),N-1);
    for j = 1:N-1
        label_train = (airport_train(:,2)==COM(i,j));
        model = fitcsvm(trainset,label_train,'KernelFunction','polynomial');
        temp(:,COM(i,j)) = predict(model,testset);
    end
    temp(:,COM(i,4)) = ones(length(testset(:,1)),1);
    temp((temp(:,COM(i,1))==1),COM(i,2)) = 0;
    temp((temp(:,COM(i,1))==1),COM(i,3)) = 0;
    temp((temp(:,COM(i,2))==1),COM(i,3)) = 0;
    temp((temp(:,COM(i,1))==1),COM(i,4)) = 0;
    temp((temp(:,COM(i,2))==1),COM(i,4)) = 0;
    temp((temp(:,COM(i,3))==1),COM(i,4)) = 0;
    
    label_test{i} = temp;
end

label_temp = zeros(length(testset(:,1)),N);
for i = 1:length(COM(:,1))
    label_temp = label_temp + label_test{i};
end
for i = 1:length(testset(:,1))
    [~,b] = sort(label_temp(i,:),'descend');
    if b(1) == b(2)
        if rand() <= 0.5
            label(i) = b(1);
        else
            label(i) = b(2);
        end
    else
        label(i) = b(1);
    end    
end

[row,~] = size(airport_sample.data);
filename = 'airport_sample_9(3).csv';
fid = fopen(filename,'w');
fprintf(fid,'%s,%s\n','Node','Class');
for i = 1:row
    fprintf(fid,'%d,%d\n',airport_sample.data(i,1),label(i));
end
