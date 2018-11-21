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
        if regr     % regressionSVM
            if best_mdl.KernelParameters.Function == "gaussian"
                data.mdl{i} = fitrsvm( train, trainLabels                   ...
                             , 'KernelFunction', best_mdl.KernelParameters.Function           ...
                             , paramGrid.paramString(1)                     ...
                                , best_mdl.KernelParameters.Scale           ...
                             , 'BoxConstraint', best_mdl.BoxConstraints(1)  ...
                             , 'Epsilon', best_mdl.Epsilon                  ...
                             );
            elseif best_mdl.KernelParameters.Function == "polynomial"
                data.mdl{i} = fitrsvm( train, trainLabels                   ...
                             , 'KernelFunction', best_mdl.KernelParameters.Function           ...
                             , paramGrid.paramString(1)                     ...
                                , best_mdl.KernelParameters.Order           ...
                             , 'BoxConstraint', best_mdl.BoxConstraints(1)  ...
                             , 'Epsilon', best_mdl.Epsilon                  ...
                             );
            end
           
            predicted = predict(data.mdl{i}, test);
            data.score{i} = calcRMSE(predicted, testLabels);
            
        else        % binarySVM
            if best_mdl.KernelParameters.Function == "gaussian"
                data.mdl{i} = fitcsvm( train, trainLabels                   ...
                             , 'KernelFunction', best_mdl.KernelParameters.Function           ...
                             , paramGrid.paramString(1)                     ...
                                , best_mdl.KernelParameters.Scale           ...
                             , 'BoxConstraint', best_mdl.BoxConstraints(1)  ...
                             );
            elseif best_mdl.KernelParameters.Function == "polynomial"
                data.mdl{i} = fitcsvm( train, trainLabels                   ...
                             , 'KernelFunction', best_mdl.KernelParameters.Function           ...
                             , paramGrid.paramString(1)                     ...
                                , best_mdl.KernelParameters.Order           ...
                             , 'BoxConstraint', best_mdl.BoxConstraints(1)  ...
                             );
            end
            
            predicted = predict(data.mdl{i}, test);
            [recall,precision] = calcRecallPrecision(predicted, testLabels);
            data.score{i} = calcF1Score(recall,precision);
        end
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
        xTest
        innerTest = xTest(innerCV.test{i}, :);
        innerTestLabels = yTest(innerCV.test{i});
        
        for m = 1:length(paramGrid.kernel)
            for j = 1:length(paramGrid.c)        
                for k = 1:length(paramGrid.kernelParam)
                    if regr     % regressionSVM
                        for l = 1:length(paramGrid.epsilon)
                            mdl = fitrsvm( innerTrain, innerTrainLabels         ...
                                , 'KernelFunction', paramGrid.kernel(m)         ...
                                , paramGrid.paramString(m), paramGrid.kernelParam(m,j)  ...
                                , 'BoxConstraint', paramGrid.c(k)               ...
                                , 'Epsilon', paramGrid.epsilon(l)               ...
                                );

                            predicted = predict(mdl, innerTest);
                            score = calcRMSE(predicted, innerTestLabels);
                            if score < best_score
                               best_mdl = mdl;
                               best_score = score;
                            end
                        end
                    else        % binarySVM
                        mdl = fitcsvm( innerTrain, innerTrainLabels         ...
                            , 'KernelFunction', paramGrid.kernel(m)         ...
                            ,  paramGrid.paramString(m), paramGrid.kernelParam(m,j) ...
                            , 'BoxConstraint', paramGrid.c(k)               ...
                            );  

                        predicted = predict(mdl, innerTest);
                        [recall,precision] = calcRecallPrecision(predicted, innerTestLabels);
                        score = calcF1Score(recall,precision);

                        if score > best_score
                           best_mdl = mdl;
                           best_score = score;
                        end
                    end
                end
            end
        end
    end
end


%{
%   Input:  Vectors containing predicted and actual labels for binary
%           classification problem.
%   Output: Recall and precision values.
%}

function [recall, precision] = calcRecallPrecision(predicted, actual)
    
    % true positive, false positive and false negative.
    tp = 0;
    fp = 0;
    fn = 0;

    % Count true positives, false positives and false negatives.
    for i = 1:length(predicted)
        if predicted(i) == 1 && actual(i) == 1
            tp = tp + 1;
        elseif predicted(i) == 1 && actual(i) == 0
            fp = fp + 1;
        elseif predicted(i) == 0 && actual(i) == 1
            fn = fn + 1;
        end
    end
    
    precision = tp / (tp + fp);
    recall    = tp / (tp + fn);
end


%{
%   Input:  Recall and precision values.
%   Output: F1 score.
%}

function f1Score = calcF1Score(recall, precision)
    f1Score = 2 * ((precision * recall) / (precision + recall));
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