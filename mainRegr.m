clc;
clear;
load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);

% Called these earlier, commented out to save time
%{
rbfParam = struct;
rbfParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
rbfParam.kernel = 'rbf';
rbfParam.paramString = 'KernelScale';
rbfParam.kernelParam = [0.001, 0.01, 0.1, 10, 100, 1000];
rbfParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
rbfRegrSVM = tuneSVMRegr(points, pose, 10, rbfParam);
save('svm/regr/rbfSVM.mat', 'rbfRegrSVM');


linearParam = struct;
linearParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
linearParam.kernel = 'linear';
% Set parameters unrelated to training so same number of arguments are 
% provided -> Can use same function to train rbf, polynomial and linear.
linearParam.paramString = 'NumPrint';
linearParam.kernelParam = 1000;
linearParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
linRegrSVM = tuneSVMRegr(points, pose, 10, linearParam);
save('svm/regr/linSVM.mat', 'linRegrSVM');


polyParam = struct;
polyParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
polyParam.kernel = "polynomial";
polyParam.paramString = "PolynomialOrder";
polyParam.kernelParam = [2 3 4];
polyParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
polRegrSVM = tuneSVMRegr(points, pose, 10, polyParam);
save('svm/regr/polSVM.mat', 'polRegrSVM');
%}

% Compare svms to ann and decision tree.
scores = comparisonRegr(points, pose, 10);

% Get mean for each vector in scores struct
means = structfun(@(x) mean(x), scores);