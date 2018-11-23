load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);

%{
regrParam = struct;
regrParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
regrParam.kernel = 'rbf';
regrParam.paramString = 'KernelScale';
regrParam.kernelParam = [0.001, 0.01, 0.1, 10, 100, 1000];
regrParam.epsilon = [0.1, 1, 10, 100];
regrSVMData = trainSVMRegr(points, pose, 10, regrParam);
%}


classParam = struct;
classParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4];
classParam.epsilon = [0.001, 0.01, 0.1, 10, 100, 1000];
classSVM = trainSVMRegr(points, pose, 10, classParam);

