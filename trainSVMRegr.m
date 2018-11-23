%   Input: paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       epsilon :: Vector of values for epsilon
%       sigma :: Vector of values for sigma
%   }
%
%   Output: data struct with format:
%   
%   struct data {
%       mdl :: SVM model
%       score :: RMSE or F1 score
%   }

function data = trainSVMRegr(X, y, k, paramGrid)
    data = struct;
    cv = partition(length(y), k);
    
    for i = 1:k
        train = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        test = X(cv.test{i}, :);
        testLabels = y(cv.test{i});
        
        % Perform grid search on first iteration of loop.
        if i == 1
            best_mdl = gridSearch(train, trainLabels, paramGrid);
            if paramGrid.kernel == "rbf"
                kernelParam = best_mdl.KernelParameters.Scale;
            elseif paramGrid.kernel == "polynomial"
                kernelParam = best_mdl.KernelParameters.Order;
            end
        end
        
        mdl = fitrsvm( train, trainLabels                          ...
                     , 'KernelFunction' , paramGrid.kernel         ...
                     , paramGrid.paramString , kernelParam         ...
                     , 'BoxConstraint', best_mdl.BoxConstraints(1) ...
                     , 'Epsilon', best_mdl.Epsilon                 ...
                     );

        predicted = predict(mdl, test);
        data.score{i} = calcRMSE(predicted, testLabels);
    end
    data.mdl = mdl;
end


function rmse = calcRMSE(predicted, actual)
    rmse = sqrt(mean(predicted - transpose(actual)).^2);
end


function best_mdl = gridSearch(X, y, paramGrid)
    
    best_score = 1000;
    for i = 1:3
        gridSearchPartition = partition(length(y), 3);
        train = X(gridSearchPartition.train{i}, :); 
        trainLabels = y(gridSearchPartition.train{i});
        test = X(gridSearchPartition.test{i}, :);
        testLabels = y(gridSearchPartition.test{i});
        
        for j = 1:length(paramGrid.c)        
            for k = 1:length(paramGrid.kernelParam)
                for l = 1:length(paramGrid.epsilon)
                    
                    mdl = fitrsvm( train, trainLabels        ...
                        , 'KernelFunction', paramGrid.kernel ...
                        , 'BoxConstraint', paramGrid.c(j)    ...
                        , paramGrid.paramString              ...
                            , paramGrid.kernelParam(k)       ...
                        , 'Epsilon', paramGrid.epsilon(l)    ...
                        );

                    predicted = predict(mdl, test);
                    score = calcRMSE(predicted, testLabels);
                    if score < best_score
                       best_mdl = mdl;
                       best_score = score;
                    end
                end
            end
        end
    end
end