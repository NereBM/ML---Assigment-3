%{
%   Input: X - Features
%          y - Labels
%          k - Number of folds
%          rbfSVM - model of rbf or gaussian SVM
%          linSVM - model of linear SVM
%          polSVM - model of polynomial SVM
%
%   Output: Struct with format:
%       scores {
%           ann :: Vector of rmse scores for neural network - length k
%           rbf :: Vector of rmse scores for rbf/gaussian svm - length k
%           lin :: Vector of rmse scores for linear svm - length k
%           pol :: Vector of rmse scores for polynomial svm - length k
%       }
%}
function scores = comparisonRegr(X, y, k)

    % Load svms previously tuned using tuneSVM*.m
    load('svm/bin/rbfSVM.mat');
    load('svm/bin/polSVM.mat');
    load('svm/bin/linSVM.mat');

    cv = partition(length(y), k);
    scores     = struct;
    scores.ann = zeros(1, k);
    scores.rbf = zeros(1, k);
    scores.lin = zeros(1, k);
    scores.pol = zeros(1, k);
    scores.tre = zeros(1, k);

    for i = 1:k
        trainData   = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        testData    = X(cv.test{i}, :);
        testLabels  = y(cv.test{i});
        
        net = feedforwardnet(10);
        net = train(net, transpose(trainData), trainLabels);  
        netPredicted = sim(net, transpose(testData));
        scores.ann(i) = calcRMSE(netPredicted, testLabels);
        
        tree = createTree(trainData, trainLabels);
        treePredicted = classify(testData, tree);
        scores.tre(i) = f1Score(treePredicted, testLabels);
        
        rbf = fitrsvm(trainData   , trainLabels                   ...
            , 'KernelFunction'    , 'rbf'                         ...
            , 'BoxConstraint'     , rbfSVM.BoxConstraints(1)      ...
            , 'KernelScale'       , rbfSVM.KernelParameters.Scale);
        rbfPredicted = transpose(predict(rbf, testData));
        scores.rbf(i) = calcRMSE(rbfPredicted, testLabels);
        
        lin = fitrsvm(trainData   , trainLabels                   ...
            , 'KernelFunction'    , 'linear'                      ...
            , 'BoxConstraint'     , linSVM.BoxConstraints(1));
        linPredicted = transpose(predict(lin, testData));
        scores.lin(i) = calcRMSE(linPredicted, testLabels);
        
        pol = fitrsvm(trainData , trainLabels                     ...
            , 'KernelFunction'  , 'polynomial'                    ...
            , 'BoxConstraint'   , polSVM.BoxConstraints(1)        ...
            , 'PolynomialOrder' , polSVM.KernelParameters.Order);
        polPredicted = transpose(predict(pol, testData));
        scores.pol(i) = calcRMSE(polPredicted, testLabels);      
    end
end