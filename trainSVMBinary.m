%   Input: paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       sigma :: Vector of values for sigma
%   }
%
%   Output: SVM with greatest f1 score.

function best_mdl = trainSVMBinary(X, y, k, paramGrid)
    cv = partition(length(y), k);
    best_score = -1;
    
    for i = 1:k
        train = X(cv.train{i}, :);
        trainLabels = y(cv.train{i});
        test = X(cv.test{i}, :);
        testLabels = y(cv.test{i});
        
        % Perform grid search on first iteration of loop.
        if i == 1
            best_mdl = nestedGridSearch(train, trainLabels, paramGrid);
            
            cp = best_mdl.BoxConstraints(1);       
            if paramGrid.kernel == "rbf"
                kp = best_mdl.KernelParameters.Scale;
            elseif paramGrid.kernel == "polynomial"
                kp = best_mdl.KernelParameters.Order;
            else
                kp = best_mdl.ModelParameters.NumPrint;
            end
        end
       
        mdl = fitcsvm( train, trainLabels                  ...
                     , 'KernelFunction' , paramGrid.kernel ...
                     , paramGrid.paramString , kp          ...
                     , 'BoxConstraint', cp                 ...
                     );
        predicted = predict(mdl, test);        
        score = f1Score(predicted, testLabels);
        
        if score > best_score
           best_mdl = mdl;
           best_score = score;
        end
    end
end


function best_mdl = nestedGridSearch(X, y, paramGrid)

    part = partition(length(y), 3);
    best_score = -1;
    
    for i = 1:3
        train = X(part.train{i}, :);
        trainLabels = y(part.train{i});
        test = X(part.test{i}, :);
        testLabels = y(part.test{i});

        for j = 1:length(paramGrid.kernelParam)
            for k = 1:length(paramGrid.c)
                fprintf("BoxConstraint = %d, Sigma/PolynomialOrder = %d\n" ...
                    , paramGrid.c(k), paramGrid.kernelParam(j));
                
                mdl = fitrsvm( train, trainLabels        ...
                    , 'KernelFunction', paramGrid.kernel ...
                    , 'BoxConstraint', paramGrid.c(k)    ...
                    , paramGrid.paramString              ...
                        , paramGrid.kernelParam(j)       ...
                    );
                
            predicted = predict(mdl, test);
            score = f1Score(predicted, testLabels);
        
            if score > best_score
               best_mdl = mdl;
               best_score = score;
            end
            end
        end
    end
end

%{
function best_mdl = single_gridSearch(X, y, paramGrid)

    part = partition(length(y), 3);
    best_score = -1;
    
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
                );
            predicted = predict(mdl, test);
            score = f1Score(predicted, actual);
        
            if score > best_score
               best_mdl = mdl;
               best_score = score;
            end
        end

        if paramGrid.kernel == "rbf" || paramGrid.kernel == "gaussian"
            kernelParam = mdl.KernelParameters.Scale;
        elseif paramGrid.kernel == "polynomial"
            kernelParam = mdl.KernelParameters.Order;
        else
            kernelParam = mdl.ModelParameters.NumPrint;
        end
        
        disp('Tuning box constraints...');
        for j = 1:length(paramGrid.c)
            mdl = fitrsvm( train, trainLabels        ...
                , 'KernelFunction', paramGrid.kernel ...
                , 'BoxConstraint', paramGrid.c(j)    ...
                , paramGrid.paramString, kernelParam ...
                );
            predicted = predict(mdl, test);
            score = f1Score(predicted, testLabels);
        
            if score > best_score
               best_mdl = mdl;
               best_score = score;
            end
        end
        
        if score > best_score
           best_mdl = mdl;
           best_score = score;
        end
    end
 end
%}