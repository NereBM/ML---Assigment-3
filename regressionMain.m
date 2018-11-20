load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                    , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);


paramGrid = struct;
paramGrid.c = [1, 10, 100, 1000];
paramGrid.kernel = 'rbf';
paramGrid.epsilon = [0.1, 1, 10, 100];

svm = regressionSVM(points, pose, 10, paramGrid);