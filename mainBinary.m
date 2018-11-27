%clear;
%clc;
load('data/facialPoints.mat');
load('data/labels.mat');
points = reshape(points, [132, 150]);
points = transpose(points);

% Called these earlier, commented out to save time

% RBF SVM
rbfParam = struct;
rbfParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
rbfParam.kernel = 'rbf';
rbfParam.paramString = 'KernelScale';
rbfParam.kernelParam = [0.001, 0.01, 0.1, 10, 50, 100, 1000];
rbfBinSVM = tuneSVMBinary(points, labels, 10, rbfParam);
save('svm/bin/rbfSVM.mat', 'rbfBinSVM');

% Polynomial SVM
polyParam = struct;
polyParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
polyParam.kernel = "polynomial";
polyParam.paramString = "PolynomialOrder";
polyParam.kernelParam = [2 3 4];
polBinSVM = tuneSVMBinary(points, labels, 10, polyParam);
save('svm/bin/polSVM.mat', 'polBinSVM');

% Linear SVM
linearParam = struct;
linearParam.c = [0.001, 0.01, 0.1, 10, 100, 1000];
linearParam.kernel = 'linear';
% Set parameters unrelated to training so same number of arguments are 
% provided -> Can use same function to train rbf, polynomial and linear.
linearParam.paramString = 'NumPrint';
linearParam.kernelParam = 1000;
linBinSVM = tuneSVMBinary(points, labels, 10, linearParam);
save('svm/bin/linSVM.mat', 'linBinSVM');

% Compare svms to ann and decision tree.
scores = comparisonBin(points, labels, 10);

% Get mean for each vector in scores struct
means = structfun(@(x) mean(x), scores);

% Number of support vectors for each SVM
svRbf = length(rbfBinSVM.IsSupportVector(rbfBinSVM.IsSupportVector == 1));
svPol = length(polBinSVM.IsSupportVector(polBinSVM.IsSupportVector == 1));
svLin = length(linBinSVM.IsSupportVector(linBinSVM.IsSupportVector == 1));

%{
rbf_pol = ttest2(scores.rbfPredicted,scores.polPredicted);
rbf_lin = ttest2(scores.rbfPredicted,scores.linPredicted);
rbf_ann = ttest2(scores.rbfPredicted,scores.annPredicted);
rbf_tre = ttest2(scores.rbfPredicted,scores.trePredicted);
pol_lin = ttest2(scores.polPredicted,scores.linPredicted); 
pol_ann = ttest2(scores.polPredicted,scores.annPredicted);
pol_tre = ttest2(scores.polPredicted,scores.trePredicted);
lin_ann = ttest2(scores.linPredicted,scores.annPredicted);
lin_tre = ttest2(scores.linPredicted,scores.trePredicted);
ann_tre = ttest2(scores.annPredicted,scores.trePredicted);
%}