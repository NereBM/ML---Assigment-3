%{
%   Input: X - Features
%          y - Labels
%          k - Number of folds
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
    load('svm/regr/rbfSVM.mat');
    load('svm/regr/polSVM.mat');
    load('svm/regr/linSVM.mat');

    cv = partition(length(y), k);
    scores     = struct;
    scores.ann = zeros(1, k);
    scores.rbf = zeros(1, k);
    scores.lin = zeros(1, k);
    scores.pol = zeros(1, k);
    
    scores.annPredicted = [];
    scores.rbfPredicted = [];
    scores.linPredicted = [];
    scores.polPredicted = [];

    for i = 1:k
        trainData   = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        testData    = X(cv.test{i}, :);
        testLabels  = y(cv.test{i});
        
        net = feedforwardnet(10);
        net.trainParam.showWindow = 0;
        net = train(net, transpose(trainData), trainLabels);  
        netPredicted  = sim(net, transpose(testData));
        scores.ann(i) = calcRMSE(netPredicted, testLabels);
        scores.annPredicted = [scores.annPredicted netPredicted];

        
        rbf = fitrsvm(trainData   , trainLabels                       ...
            , 'KernelFunction'    , 'rbf'                             ...
            , 'BoxConstraint'     , rbfRegrSVM.BoxConstraints(1)      ...
            , 'KernelScale'       , rbfRegrSVM.KernelParameters.Scale ...
            , 'Epsilon'           , rbfRegrSVM.Epsilon);
        rbfPredicted = transpose(predict(rbf, testData));
        scores.rbf(i) = calcRMSE(rbfPredicted, testLabels);
        scores.rbfPredicted = [scores.rbfPredicted rbfPredicted];
        
        lin = fitrsvm(trainData   , trainLabels                       ...
            , 'KernelFunction'    , 'linear'                          ...
            , 'BoxConstraint'     , linRegrSVM.BoxConstraints(1)      ...
            , 'Epsilon'           , linRegrSVM.Epsilon);
        linPredicted = transpose(predict(lin, testData));
        scores.lin(i) = calcRMSE(linPredicted, testLabels);
        scores.linPredicted = [scores.linPredicted linPredicted];
        
        pol = fitrsvm(trainData , trainLabels                         ...
            , 'KernelFunction'  , 'polynomial'                        ...
            , 'BoxConstraint'   , polRegrSVM.BoxConstraints(1)        ...
            , 'PolynomialOrder' , polRegrSVM.KernelParameters.Order   ...
            , 'Epsilon'         , polRegrSVM.Epsilon);
        polPredicted = transpose(predict(pol, testData));
        scores.pol(i) = calcRMSE(polPredicted, testLabels); 
        scores.polPredicted = [scores.polPredicted polPredicted];
    end
end

function rmse = calcRMSE(predicted, actual)
    rmse = sqrt(mean(predicted - actual).^2);
end