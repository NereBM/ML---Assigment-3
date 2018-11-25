load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);

%{
rbfParam = struct;
rbfParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
rbfParam.kernel = 'rbf';
rbfParam.paramString = 'KernelScale';
rbfParam.kernelParam = [0.001, 0.01, 0.1, 10, 100, 1000];
rbfParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
rbf_regr_svm = tuneSVMRegr(points, pose, 10, rbfParam);
%}

%{
linearParam = struct;
linearParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
linearParam.kernel = 'linear';
% Set parameters unrelated to training so same number of arguments are 
% provided -> Can use same function to train rbf, polynomial and linear.
linearParam.paramString = 'NumPrint';
linearParam.kernelParam = 1000;
linearParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
linear_regr_svm = tuneSVMRegr(points, pose, 10, linearParam);
%}

%{
polyParam = struct;
polyParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
polyParam.kernel = "polynomial";
polyParam.paramString = "PolynomialOrder";
polyParam.kernelParam = [2 3 4];
polyParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
poly_regr_svm = tuneSVMRegr(points, pose, 10, polyParam);
%}