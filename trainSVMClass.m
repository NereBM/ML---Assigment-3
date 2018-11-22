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

function data = trainSVMClass(X, y, k, paramGrid)
    data = struct;
    cv = partition(length(y), k);
    
    for i = 1:k
        train = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        test = X(cv.test{i}, :);
        testLabels = y(cv.test{i});
        
        % Perform grid search on each iteration of loop.
        best_mdl = gridSearch(train, trainLabels, paramGrid);
        if paramGrid.kernel == "rbf"
            kernelParam = best_mdl.BinaryLearners{1,1}.KernelParameters.Scale;  % finds the best scale from the model
        elseif paramGrid.kernel == "polynomial"
            kernelParam = best_mdl.BinaryLearners{1,1}.KernelParameters.Order;  % finds the best order from the model
        end
        
        t = templateSVM('Standardize',1,'KernelFunction',paramGrid.kernel ...
                    , paraGrid.paramString, kernelParam                   ...
                    , 'BoxConstraint',best_mdl.BoxConstraints(1));  

        %%%% The attempt to get the box constraint from above is where it
        %%%% crashes, unable to find the box constraint in the best_mdl
        
        mdl = fitcecoc( train, trainLabels                          ...
                    , 'Learners',t );
        
                 
        predicted = predict(mdl, test);        
        [recall, precision] = calcRecallPrecision(predicted, testLabels);
        data.score{i} = calcF1Score(recall, precision);
        data.mdl{i} = mdl;
    end
end
        
function best_mdl = gridSearch(X, y, paramGrid)

    gridSearchPartition = partition(length(y), 3);
    best_score = 0;
    for i = 1:3
        train = X(gridSearchPartition.train{i}, :);
        trainLabels = y(gridSearchPartition.train{i});
        test = X(gridSearchPartition.test{i}, :);
        testLabels = y(gridSearchPartition.test{i});
        
        for j = 1:length(paramGrid.c)        
            for k = 1:length(paramGrid.kernelParam)
                    
                
                t = templateSVM('Standardize',1,                        ...
                    'KernelFunction',paramGrid.kernel                   ...
                    , paramGrid.paramString, paramGrid.kernelParam(k)    ...
                    , 'BoxConstraint',paramGrid.c(j));

                mdl = fitcecoc( train, trainLabels                ...
                    , 'Learners',t );
                

                predicted = predict(mdl, test);
                confusion_matrix = confusion_matrix_generator(predicted,testLabels,6);
                score = calcF1Score(confusion_matrix);

                if score > best_score
                   best_mdl = mdl;
                   best_score = score;
                end
            end
        end
    end
end

%{
%   Input:  Vectors containing predicted and actual labels for multiclass
%           classification problem.
%   Output: A confusion matrix for the results
%}

function [confusion_matrix] = confusion_matrix_generator(predicted_results,actual_results,size_of_matrix)
% This function assumes that we are given two matrices containing the
% values of the actual results and the predicted results. It will then
% determine the resulting confusion matrix.

confusion_matrix = zeros(size_of_matrix);

        for i=1:length(predicted_results)
            confusion_matrix(actual_results(i), predicted_results(i)) = confusion_matrix(actual_results(i), predicted_results(i)) + 1;
        end
end

%{
%   Input:  A confusion matrix
%
%   Output: The F1 score for the given confusion matrix
%}

function f1Score = calcF1Score(confusion_matrix)
    F1Sum = 0;
    for i=1:length(confusion_matrix)
        TP = confusion_matrix(i,i);
        FP = sum(confusion_matrix(:,i)) - TP;
        FN = sum(confusion_matrix(i,:)) - TP;
        
        PRE = TP / (TP + FP);
        REC = TP / (FN + TP);
        F1 = 2 * (PRE*REC)/(PRE+REC);
        F1Sum = F1Sum + F1;
    end
    f1Score = F1Sum/length(confusion_matrix);
end