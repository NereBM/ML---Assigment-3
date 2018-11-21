clear
load('binaryData/facialPoints.mat');
load('binaryData/labels.mat');
binaryPoints = reshape(points, [132, 150]);
binaryLabels = transpose(labels);
binaryPoints = transpose(binaryPoints);

paramGrid = struct;
paramGrid.c = [0.1, 1, 10, 100];
paramGrid.kernel = ["rbf","polynomial"];
paramGrid.paramString = ["KernelScale","PolynomialOrder"];
paramGrid.kernelParam = [0.1, 1, 10, 100 ; 2 3 4 5];
svmBinary = regressionSVM(binaryPoints, binaryLabels, 10, paramGrid, false);





