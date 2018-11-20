% paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       epsilon :: Vector of values for epsilon

function mdls = regressionSVM(X, y, k, paramGrid)
    cv = partition(length(y), k);
    mdls.mdl = {};
    mdls.rmse = [];
    
    
    for i = 1:k
        outerTrain = X(cv.train{i}, :);
        outerTrainActual = y(cv.train{i});
        outerTest = X(cv.test{i}, :);
        outerTestActual = y(cv.test{i});
        
        for j = 1:3
            innerCV = partition(size(outerTrain, 2), 3);
            innerTrain = outerTrain(innerCV.train{j}, :);
            innerTrainActual = outerTrainActual(innerCV.train{j});
            innerTest = outerTest(innerCV.test{j}, :);
            innerTestActual = outerTestActual(innerCV.test{j});
            
            % Perform grid search
            for k = 1:1
                for l = 1:1
                    % tune parameters within inner loop
                    mdls.mdl{i*j*k*l} =                             ...
                        fitrsvm( innerTrain, innerTrainActual       ...
                               , 'BoxConstraint', paramGrid.c(k)    ...
                               , 'KernelFunction', paramGrid.kernel ...
                               , 'Epsilon', paramGrid.epsilon(l)    ...
                               );
                    predicted = predict(mdls.mdl{i*j*k*l}, innerTest);
                    calcRMSE(predicted, innerTestActual)
                    mdls.rmse(i*j*k*l) = calcRMSE(predicted, innerTestActual);
                end
            end
        end
    end
end

function rmse = calcRMSE(predicted, actual)
    rmse = sqrt(mean((transpose(predicted) - actual).^2));
end
%{
%   Optimal hyper-parameters to find:
%       'BoxConstraint' - for all
%           - Positive scalar value
%       'KernelScale' - for classification
%           - Positive scalar value
%       'Epsilon' - for regression
%           - Non negative scalar value
%       'sigma' - for regression
%           - Non negative scalar value
%
%   SEE: https://www.mathworks.com/help/stats/fitrsvm.html#namevaluepairs
%}