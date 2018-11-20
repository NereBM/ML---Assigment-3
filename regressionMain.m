load('data/regressFacialPoints.mat');
load('data/regressHeadpose.mat');

points = reshape(points, ...
                [size(points, 1) * size(points, 2) ...
                    , size(points, 3)]);
pose = transpose(pose(:,6));
points = transpose(points);

svm = regressionSVM(points, pose, 'rbf', 0.4);