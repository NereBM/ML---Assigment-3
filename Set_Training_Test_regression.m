load('data/regressfacialPoints.mat');
load('data/regressHeadPose.mat');

S = size(points);
PointsT = reshape(points,[S(1)*S(2),S(3)]);%Compact the data

labels = pose(:,6);
LabelsT = labels';
data_matrix=vertcat(PointsT,LabelsT);
data_matrix_per=data_matrix(:,randperm(size(data_matrix,2)));

regress_test_features = data_matrix_per(1:132,1:floor(size(data_matrix_per,2)/5));
regress_test_labels = data_matrix_per(133,1:floor(size(data_matrix_per,2)/5));
regress_training_features = data_matrix_per(1:132,(floor(size(data_matrix_per,2)/5)+1):end);
regress_training_labels = data_matrix_per(133,(floor(size(data_matrix_per,2)/5)+1):end);
save('regress_test_features.mat','regress_test_features')
save('regress_training_features.mat','regress_training_features')
save('regress_training_labels.mat','regress_training_labels')
save('regress_test_labels.mat','regress_test_labels')
