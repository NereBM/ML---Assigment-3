%   Input: paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       sigma :: Vector of values for sigma
%   }
%
%   Output: data struct with format:
%   
%   struct data {
%       mdl :: SVM model
%       score :: RMSE or F1 score
%   }

function data = binarySVM(X, y, k, paramGrid)
    data = struct;
    cv = partition(length(y), k);
   
    for i = 1:k     % inner-fold cross validation for hyper-parameter optimisation
        train = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        
        % Perform grid search on each iteration of the loop tuning the
        % parameterspolySVMdata
        if i == 1
            [best_mdl,best_score] = gridSearch(train, trainLabels, paramGrid, 0, 0);
        else
            [best_mdl,best_score] = gridSearch(train, trainLabels, paramGrid, best_mdl, best_score);
        end
        
        if paramGrid.kernel == "rbf"
            kernelParam = best_mdl.KernelParameters.Scale;
        elseif paramGrid.kernel == "polynomial"
            kernelParam = best_mdl.KernelParameters.Order;
        end
    
    
        train = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        test = X(cv.test{i}, :);
        testLabels = y(cv.test{i});
        
        mdl = fitcsvm( train, trainLabels                          ...
                         , 'KernelFunction' , paramGrid.kernel         ...
                         , paramGrid.paramString , kernelParam         ...
                         , 'BoxConstraint', best_mdl.BoxConstraints(1) ...
                         );


        predicted = predict(mdl, test);
        transpose(predicted)
        testLabels
        [recall, precision] = calcRecallPrecision(predicted, testLabels)
        data.score{i} = calcF1Score(recall, precision)
        data.mdl{i} = mdl;
    end
end
        
function [best_mdl,best_score] = gridSearch(X, y, paramGrid,     ...
                                        previousBestMdl,previousBestScore)
                       
    
    best_score = previousBestScore;
    best_mdl = previousBestMdl;

    gridSearchPartition = partition(length(y), 3);
    % we only need to use the first partition of this partition function
    
    train = X(gridSearchPartition.train{1},:);
    trainLabels = y(gridSearchPartition.train{1});
    test = X(gridSearchPartition.test{1}, :);
    testLabels = y(gridSearchPartition.train{1});
    
    for j = 1:length(paramGrid.c)        
        for k = 1:length(paramGrid.kernelParam)

            mdl = fitcsvm( train, trainLabels        ...
                , 'KernelFunction', paramGrid.kernel ...
                , 'BoxConstraint', paramGrid.c(j)    ...
                , paramGrid.paramString              ...
                    , paramGrid.kernelParam(k)       ...
                );

            predicted = predict(mdl, test);
            [recall, precision] = calcRecallPrecision(predicted, testLabels);
            score = calcF1Score(recall, precision);

            if score > best_score
               best_mdl = mdl;
               best_score = score;
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

function f1Score = calcF1Score(recall, precision)
    f1Score = 2 * ((precision * recall) / (precision + recall));
end