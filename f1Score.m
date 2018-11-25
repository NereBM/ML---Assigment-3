function f1Score = f1Score(predicted, actual)
    [recall, precision] = calcRecallPrecision(predicted, actual);
    f1Score = calcF1Score(recall, precision);
end

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