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
function scores = comparisonBin(X, y, k)

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
        net = train(net, transpose(trainData), transpose(trainLabels));  
        netPredicted = round(sim(net, transpose(testData)));
        scores.ann(i) = f1Score(netPredicted, testLabels);
        
        tree = createTree(transpose(trainData), transpose(trainLabels));
        treePredicted = classify(transpose(testData), tree);
        scores.tre(i) = f1Score(treePredicted, testLabels);
        
        rbf = fitcsvm(trainData   , trainLabels                   ...
            , 'KernelFunction'    , 'rbf'                         ...
            , 'BoxConstraint'     , rbfBinSVM.BoxConstraints(1)   ...
            , 'KernelScale'       , rbfBinSVM.KernelParameters.Scale);
        rbfPredicted = transpose(predict(rbf, testData));
        scores.rbf(i) = f1Score(rbfPredicted, testLabels);
        
        lin = fitcsvm(trainData   , trainLabels                   ...
            , 'KernelFunction'    , 'linear'                      ...
            , 'BoxConstraint'     , linBinSVM.BoxConstraints(1));
        linPredicted = transpose(predict(lin, testData));
        scores.lin(i) = f1Score(linPredicted, testLabels);
        
        pol = fitcsvm(trainData , trainLabels                        ...
            , 'KernelFunction'  , 'polynomial'                       ...
            , 'BoxConstraint'   , polBinSVM.BoxConstraints(1)        ...
            , 'PolynomialOrder' , polBinSVM.KernelParameters.Order);
        polPredicted = transpose(predict(pol, testData));
        scores.pol(i) = f1Score(polPredicted, testLabels);      
    end
end