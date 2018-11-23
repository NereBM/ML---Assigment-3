clear
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
rbfSVMData = trainSVMBinary(binaryPoints, binaryLabels, 10, classParam);
%}
%{
classParam = struct;
classParam.c = [0.001,0.01, 0.1, 1, 10, 100, 1000];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4];
polySVMdata = trainSVMBinary(binaryPoints, binaryLabels, 10, classParam);
%}

classParam = struct;
classParam.c = [0.001,0.01, 0.1, 1, 10, 100, 1000];
classParam.kernel = "linear";
linearSVMdata = trainSVMBinary(binaryPoints, binaryLabels, 10, classParam);

