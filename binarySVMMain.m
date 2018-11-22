clear
load('binaryData/facialPoints.mat');
load('binaryData/labels.mat');
binaryPoints = reshape(points, [132, 150]);
binaryLabels = transpose(labels);
binaryPoints = transpose(binaryPoints);


classparam = struct;
classParam.c = [0.1, 1, 10, 100, 500 1000];
classParam.kernel = 'rbf';
classParam.paramString = 'KernelScale';
classParam.kernelParam = [0.1, 1, 10, 100, 500];
rbfSVMData = binarySVM(binaryPoints, binaryLabels, 10, classParam);

%{
classParam = struct;
classParam.c = [0.01, 0.1, 1, 10, 100, 500];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4 5 6 7];
polySVMdata = binarySVM(binaryPoints, binaryLabels, 10, classParam);
%}
