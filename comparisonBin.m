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
    
    scores.annPredicted = [];
    scores.rbfPredicted = [];
    scores.linPredicted = [];
    scores.polPredicted = [];
    scores.trePredicted = [];
    

    for i = 1:k
        trainData   = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        testData    = X(cv.test{i}, :);
        testLabels  = y(cv.test{i});
        
        net = feedforwardnet(10);
        net.trainParam.showWindow = 0;
        net = train(net, transpose(trainData), transpose(trainLabels));  
        netPredicted = round(sim(net, transpose(testData)));
        scores.ann(i) = f1Score(netPredicted, testLabels);
        scores.annPredicted = [scores.annPredicted netPredicted];
        
        tree = createTree(transpose(trainData), transpose(trainLabels));
        treePredicted = classify(transpose(testData), tree);
        scores.tre(i) = f1Score(treePredicted, testLabels);
        scores.trePredicted = [scores.trePredicted treePredicted'];
        
        rbf = fitcsvm(trainData   , trainLabels                   ...
            , 'Standardize', true                                     ...
            , 'KernelFunction'    , 'rbf'                         ...
            , 'BoxConstraint'     , rbfBinSVM.BoxConstraints(1)   ...
            , 'KernelScale'       , rbfBinSVM.KernelParameters.Scale);
        rbfPredicted = transpose(predict(rbf, testData));
        scores.rbf(i) = f1Score(rbfPredicted, testLabels);
        scores.rbfPredicted = [scores.rbfPredicted rbfPredicted];

        lin = fitcsvm(trainData   , trainLabels                   ...
            , 'Standardize', true                                     ...
            , 'KernelFunction'    , 'linear'                      ...
            , 'BoxConstraint'     , linBinSVM.BoxConstraints(1));
        linPredicted = transpose(predict(lin, testData));
        scores.lin(i) = f1Score(linPredicted, testLabels);
        scores.linPredicted = [scores.linPredicted linPredicted];
        
        pol = fitcsvm(trainData , trainLabels                        ...
            , 'Standardize', true                                     ...
            , 'KernelFunction'  , 'polynomial'                       ...
            , 'BoxConstraint'   , polBinSVM.BoxConstraints(1)        ...
            , 'PolynomialOrder' , polBinSVM.KernelParameters.Order);
        polPredicted = transpose(predict(pol, testData));
        scores.pol(i) = f1Score(polPredicted, testLabels); 
        scores.polPredicted = [scores.polPredicted polPredicted];

    end
end