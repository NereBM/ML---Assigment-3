load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                    , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);


paramGrid = struct;
paramGrid.c = [0.1, 1, 10, 100];
paramGrid.kernel = ["rbf","polynomial"];
paramGrid.paramString = ["KernelScale","PolynomialOrder"];
paramGrid.kernelParam = [0.1, 1, 10, 100 ; 2 3 4 5];
paramGrid.epsilon = [0.1, 1, 10, 100];
svm = regressionSVM(points, pose, 10, paramGrid, true);



