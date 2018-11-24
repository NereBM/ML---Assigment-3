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
rbf_regr_SVM = trainSVMRegr(points, pose, 10, rbfParam);
%}

%{
polyParam = struct;
polyParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
polyParam.kernel = "polynomial";
polyParam.paramString = "PolynomialOrder";
polyParam.kernelParam = [1 2 3 4];
polyParam.epsilon = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
poly_regr_SVM = trainSVMRegr(points, pose, 10, polyParam);
%}