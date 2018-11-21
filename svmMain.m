load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);


regrParam = struct;
regrParam.c = [0.1, 1, 10, 100];
regrParam.kernel = 'rbf';
regrParam.paramString = 'KernelScale';
regrParam.kernelParam = [0.1, 1, 10, 100];
regrParam.epsilon = [0.1, 1, 10, 100];
regrSVMData = trainSVMRegr(points, pose, 10, regrParam);

%{
classParam = struct;
classParam.c = [0.1, 1, 10, 100];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4 5];
classParam.epsilon = [0.1, 1, 10, 100];
classSVM = trainSVM(points, pose, 10, classParam, true);
%}
