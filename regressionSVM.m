% paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       epsilon :: Vector of values for epsilon
%       sigma :: Vector of values for sigma


% struct data {
%     mdl :: SVM model
%     score :: RMSE or F1 score
% }

function data = regressionSVM(X, y, k, paramGrid, regr)
    cv = partition(length(y), k);
    
    for i = 1:k
        train = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        test = X(cv.test{i}, :);
        testLabels = y(cv.test{i});
        
        % Perform grid search on first iteration of loop.
        if i == 1
            best_mdl = gridSearch( train, trainLabels ...
                            , test,  testLabels ...
                            , paramGrid, regr);
        end
        
        data.mdl{i} = fitrsvm( train, trainLabels                   ...
                     , 'KernelFunction', paramGrid.kernel           ...
                     , paramGrid.paramString                        ...
                        , best_mdl.KernelParameters.Scale           ...
                     , 'BoxConstraint', best_mdl.BoxConstraints(1)  ...
                     , 'Epsilon', best_mdl.Epsilon                  ...
                     );
        predicted = predict(data.mdl{i}, test);
        data.score{i} = calcRMSE(predicted, testLabels);
    end
end

function rmse = calcRMSE(predicted, actual)
    rmse = sqrt(mean(predicted - transpose(actual)).^2);
end

function best_mdl = gridSearch(xTrain, yTrain, xTest, yTest, paramGrid, regr)
    
    best_score = 1000;

    for i = 1:3
        innerCV = partition(size(xTrain, 2), 3);
        innerTrain = xTrain(innerCV.train{i}, :);
        innerTrainLabels = yTrain(innerCV.train{i});
        innerTest = xTest(innerCV.test{i}, :);
        innerTestLabels = yTest(innerCV.test{i});

        for j = 1:length(paramGrid.c)        
            for k = 1:length(paramGrid.kernelParam)
                if regr
                    for l = 1:length(paramGrid.epsilon)
                        mdl = fitrsvm( innerTrain, innerTrainLabels     ...
                            , 'KernelFunction', paramGrid.kernel        ...
                            , paramGrid.paramString, paramGrid.kernelParam(j)  ...
                            , 'BoxConstraint', paramGrid.c(k)           ...
                            , 'Epsilon', paramGrid.epsilon(l)           ...
                            );

                        predicted = predict(mdl, innerTest);
                        score = calcRMSE(predicted, innerTestLabels);
                        if score < best_score
                           best_mdl = mdl;
                           best_score = score;
                        end
                    end
                else
                    mdl = fitcsvm( innerTrain, innerTrainLabels     ...
                        , 'KernelFunction', paramGrid.kernel        ...
                        ,  paramGrid.paramString, paramGrid.kernelParam(j) ...
                        , 'BoxConstraint', paramGrid.c(k)           ...
                        );  
                    
                    predicted = predict(mdl, innerTest);
                    score = calcF1(predicted, innerTestLabels);
                    if score > best_score
                       best_mdl = mdl;
                       best_score = score;
                    end
                end
            end
        end
    end
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