clear;
clc;
load('binaryData/facialPoints.mat');
load('binaryData/labels.mat');
binaryPoints = reshape(points, [132, 150]);
binaryLabels = transpose(labels);
binaryPoints = transpose(binaryPoints);

%{
classParam = struct;
classParam.c = [0.001, 0.01,0.1, 1, 10, 100, 500 1000];
classParam.kernel = 'rbf';
classParam.paramString = 'KernelScale';
classParam.kernelParam = [0.1, 1, 10, 100, 500];
rbfBinSVM = trainSVMBinary(binaryPoints, binaryLabels, 10, classParam);
%}


classParam = struct;
classParam.c = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4];
polyBinSVM = trainSVMBinary(binaryPoints, binaryLabels, 10, classParam);


%{
classParam = struct;
classParam.c = [0.001,0.01, 0.1, 1, 10, 100, 1000];
classParam.kernel = "linear";
% Set parameters unrelated to training so same number of arguments are 
% provided -> Can use same function to train rbf, polynomial and linear.
classParam.paramString = 'NumPrint';
classParam.kernelParam = 1000;
linBinSVM = trainSVMBinary(binaryPoints, binaryLabels, 10, classParam);
%}