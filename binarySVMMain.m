clear
load('binaryData/facialPoints.mat');
load('binaryData/labels.mat');
binaryPoints = reshape(points, [132, 150]);
binaryLabels = transpose(labels);
binaryPoints = transpose(binaryPoints);


classParam = struct;
classParam.c = [0.1, 1, 10, 100];
classParam.kernel = 'rbf';
classParam.paramString = 'KernelScale';
classParam.kernelParam = [0.1, 1, 10, 100];
rbfSVMData = trainSVMClass(binaryPoints, binaryLabels, 10, classParam);


classParam = struct;
classParam.c = [0.1, 1, 10, 100];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4 5];
polySVMdata = trainSVMClass(binaryPoints, binaryLabels, 10, classParam);

