clear;
clc;
load('data/facialPoints.mat');
load('data/labels.mat');
points = reshape(points, [132, 150]);
points = transpose(points);

classParam = struct;
classParam.c = [1 10 100 500 1000];
classParam.kernel = 'rbf';
classParam.paramString = 'KernelScale';
classParam.kernelParam = [0.001 0.01 0.1 50 100];
rbfBinSVM = tuneSVMBinary(points, labels, 10, classParam);

%{
classParam = struct;
classParam.c = [1, 10, 100, 1000];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4];
polyBinSVM = tuneSVMBinary(binaryPoints, binaryLabels, 10, classParam);
%}

%{
classParam = struct;
classParam.c = [1, 10, 100, 1000];
classParam.kernel = "linear";
% Set parameters unrelated to training so same number of arguments are 
% provided -> Can use same function to train rbf, polynomial and linear.
classParam.paramString = 'NumPrint';
classParam.kernelParam = 1000;
linBinSVM = tuneSVMBinary(binaryPoints, binaryLabels, 10, classParam);
%}