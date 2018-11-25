function f1Score = calcF1Score(recall, precision)
    f1Score = 2 * ((precision * recall) / (precision + recall));
end