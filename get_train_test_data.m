%数据根目录
rootFolder = 'F:\imagenetdata\101_ObjectCategories';
%数据分类
categories = {'airplanes','ferry','laptop'};
%保存数据路径
imds = imageDatastore(fullfile(rootFolder,categories),...
    'labelSource','foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds,minSetCount,'randomize');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:67
    tmp = imread(imds.Files{i});
    tmp = imresize(tmp,[128,128]);
    if sum(size(tmp)) ~= 259
        tmp = repmat(tmp,[1,1,3]);
    end
    image_data(i,:,:,:) = tmp;
    image_data_label(i,:) = [1 0 0];
end
for i=1:67
    tmp = imrotate(reshape(image_data(i,:,:,:),[128,128,3]),90);
    image_data(i+67,:,:,:) = tmp;
    image_data_label(i+67,:) = [1 0 0];
end

for i=1:67
    tmp = imrotate(reshape(image_data(i,:,:,:),[128,128,3]),180);
    image_data(i+67*2,:,:,:) = tmp;
    image_data_label(i+67*2,:) = [1 0 0];
end

for i=1:67
    tmp = imrotate(reshape(image_data(i,:,:,:),[128,128,3]),-90);
    image_data(i+67*3,:,:,:) = tmp;
    image_data_label(i+67*3,:) = [1 0 0];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%选择训练集数据和测试集数据
for i=1:67
    tmp = imread(imds.Files{67+i});
    tmp = imresize(tmp,[128,128]);
    if sum(size(tmp)) ~= 259
        tmp = repmat(tmp,[1,1,3]);
    end
    image_data(i+67*4,:,:,:) = tmp;
    image_data_label(i+67*4,:) = [0 1 0];
end
for i=1:67
    tmp = imrotate(reshape(image_data(i+67*4,:,:,:),[128,128,3]),90);
    image_data(i+67*5,:,:,:) = tmp;
    image_data_label(i+67*5,:) = [0 1 0];
end

for i=1:67
    tmp = imrotate(reshape(image_data(i+67*4,:,:,:),[128,128,3]),180);
    image_data(i+67*6,:,:,:) = tmp;
    image_data_label(i+67*6,:) = [0 1 0];
end

for i=1:67
    tmp = imrotate(reshape(image_data(i+67*4,:,:,:),[128,128,3]),-90);
    image_data(i+67*7,:,:,:) = tmp;
    image_data_label(i+67*7,:) = [0 1 0];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:67
    tmp = imread(imds.Files{67*2+i});
    tmp = imresize(tmp,[128,128]);
    if sum(size(tmp)) ~= 259
        tmp = repmat(tmp,[1,1,3]);
    end
    image_data(i+67*8,:,:,:) = tmp;
    image_data_label(i+67*8,:) = [0 0 1];
end
for i=1:67
    tmp = imrotate(reshape(image_data(i+67*8,:,:,:),[128,128,3]),90);
    image_data(i+67*9,:,:,:) = tmp;
    image_data_label(i+67*9,:) = [0 0 1];
end

for i=1:67
    tmp = imrotate(reshape(image_data(i+67*8,:,:,:),[128,128,3]),180);
    image_data(i+67*10,:,:,:) = tmp;
    image_data_label(i+67*10,:) = [0 0 1];
end

for i=1:67
    tmp = imrotate(reshape(image_data(i+67*8,:,:,:),[128,128,3]),-90);
    image_data(i+67*11,:,:,:) = tmp;
    image_data_label(i+67*11,:) = [0 0 1];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% num_imds = sum(tbl{:,2});
% idx = randperm(num_imds);
%将训练数据保存在四维张量image_train中,训练数据标签保存在image_train_label中
%shuffle数据
idx = randperm(804);
idx_train = idx(1:700);
idx_test = idx(701:804);
image_train = image_data(idx_train,:,:,:);
image_train_label = image_data_label(idx_train,:);
%将训练数据保存在四维张量image_test中,训练数据标签保存在image_test_label中
image_test = image_data(idx_test,:,:,:);
image_test_label = image_data_label(idx_test,:);