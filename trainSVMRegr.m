%   Input: paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       epsilon :: Vector of values for epsilon
%       sigma :: Vector of values for sigma
%   }
%
%   Output: SVM with lowest rmse.

function best_mdl = trainSVMRegr(X, y, k, paramGrid)
    best_score = 1000;
    cv = partition(length(y), k);
    
    for i = 1:k
        if i == 1
            best_mdl = gridSearch(X(cv.train{i}, :), y(cv.train{i}), paramGrid);           
            if paramGrid.kernel == "rbf" || paramGrid.kernel == "gaussian"
                kernelParam = best_mdl.KernelParameters.Scale;
            elseif paramGrid.kernel == "polynomial"
                kernelParam = best_mdl.KernelParameters.Order;
            else
                kernelParam = best_mdl.ModelParameters.NumPrint;
            end
        end
        mdl = fitrsvm( X(cv.train{i}, :), y(cv.train{i})           ...
                     , 'KernelFunction' , paramGrid.kernel         ...
                     , paramGrid.paramString , kernelParam         ...
                     , 'BoxConstraint', best_mdl.BoxConstraints(1) ...
                     , 'Epsilon', best_mdl.Epsilon                 ...
                     );
        predicted = predict(mdl, X(cv.test{i}, :));
        score = calcRMSE(predicted, y(cv.test{i}));
        
        if score < best_score
           best_score = score;
           best_mdl = mdl;
        end
    end
end

% Tune parameters one at a time for O(3*n) instead of O(3^n) as
% per email with Tom.
function best_mdl = gridSearch(X, y, paramGrid)

    part = partition(length(y), 3);
    best_score = 1000;
    
    for i = 1:3
        train = X(part.train{i}, :);
        trainLabels = y(part.train{i});
        test = X(part.test{i}, :);
        testLabels = y(part.test{i});
             
        disp('Tuning sigma/polynomial order...');
        for j = 1:length(paramGrid.kernelParam)
            mdl = fitrsvm( train, trainLabels        ...
                , 'KernelFunction', paramGrid.kernel ...
                , 'BoxConstraint', paramGrid.c(1)    ...
                , paramGrid.paramString              ...
                    , paramGrid.kernelParam(j)       ...
                , 'Epsilon', paramGrid.epsilon(1)    ...
                );
            predicted = predict(mdl, test);
            score = calcRMSE(predicted, testLabels);
        
            if score < best_score
               best_mdl = mdl;
               best_score = score;
            end
        end

        if paramGrid.kernel == "rbf" || paramGrid.kernel == "gaussian"
            kernelParam = best_mdl.KernelParameters.Scale;
        elseif paramGrid.kernel == "polynomial"
            kernelParam = best_mdl.KernelParameters.Order;
        else
            kernelParam = best_mdl.ModelParameters.NumPrint;
        end
        
        disp('Tuning epsilon...');
        for j = 1:length(paramGrid.epsilon)
            mdl = fitrsvm( train, trainLabels        ...
                , 'KernelFunction', paramGrid.kernel ...
                , 'BoxConstraint', paramGrid.c(1)    ...
                , paramGrid.paramString, kernelParam ...
                , 'Epsilon', paramGrid.epsilon(j)    ...
                );
            predicted = predict(mdl, test);
            score = calcRMSE(predicted, testLabels);
        
            if score < best_score
               best_mdl = mdl;
               best_score = score;
            end
        end
        
        disp('Tuning box constraints...');
        for j = 1:length(paramGrid.c)
            mdl = fitrsvm( train, trainLabels        ...
                , 'KernelFunction', paramGrid.kernel ...
                , 'BoxConstraint', paramGrid.c(j)    ...
                , paramGrid.paramString, kernelParam ...
                , 'Epsilon', best_mdl.Epsilon        ...
                );
            predicted = predict(mdl, test);
            score = calcRMSE(predicted, testLabels);
        
            if score < best_score
               best_mdl = mdl;
               best_score = score;
            end
        end
        
        if score < best_score
           best_mdl = mdl;
           best_score = score;
        end
    end
end