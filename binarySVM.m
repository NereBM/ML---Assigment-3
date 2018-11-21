% paramGrid is struct with format:
%
%   struct paramGrid {
%       c :: Vector of values for c
%       kernel :: String 
%       epsilon :: Vector of values for epsilon

function mdls = binarySVM(X, y, k, paramGrid)
    cv = partition(length(y), k);
    mdls.mdl = {};
    mdls.f1Score = [];
    
    
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
            for k = 1:(length(paramGrid.c)
                for l = 1:(length(paramGrid.sigma))
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