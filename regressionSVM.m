function mdl = regressionSVM(X, y, kernelF, epsilonValue)

    cv = partition(length(y), 10);

    for i = 1:10
        outerTrain = X(cv.train{i}, :);
        outerTest = y(cv.train{i});

        for j = 1:3
            innerCV = partition(length(outerTrain), 3);
            innerTrain = outerTrain(innerCV.train{j}, :);
            innerTest = outerTest(innerCV.train{j});

            % tune parameters within inner loop
            mdl = fitrsvm( innerTrain       ...
                         , innerTest        ...
                         , 'KernelFunction' ...
                         , kernelF          ...
                         , 'Epsilon'        ...
                         , epsilonValue); 
        end
    end
end
%{
%   Optimal hyper-parameters to find:
%       'BoxConstraint' - for classification
%           - Positive scalar value
%       'KernelScale' - for classification
%           - Positive scalar value
%       'Epsilon' - for regression
%           - Non negative scalar value
%
%   SEE: https://www.mathworks.com/help/stats/fitrsvm.html#namevaluepairs
%}