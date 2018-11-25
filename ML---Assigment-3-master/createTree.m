%{
%   Input:  Matrix of features, vector of corresponding labels.
%   Output: Struct containing decision tree with format:
%
%   tree {
%       op        :: String, index of attribute tested.
%       attribute :: Number, index of attribute tested.
%       threshold :: Number, used to split data.
%       class     :: Number, 1 or 0.
%       kids      :: Cell array of length 2 containing subtrees as structs.
%       entropy   :: Number, entropy value at node.
%  }
%}

function tree = createTree(features, labels)
    tree.entropy = calcEntropy(labels);
    tree.kids = {};
    tree.op = '';
    
    % This doesn't make sense. If features is empty, labels will be too.
    if isempty(features)
        tree.class = majorityVote(labels);
    elseif sum(labels(1) == labels) == length(labels)
        tree.class = labels(1);
    else 
        % Function to select best attribute and threshold to split data.
        [tree.attribute, tree.threshold] = ...
            choose_attributes(features, transpose(labels));
        tree.op = int2str(tree.attribute);
               
        % Divide data in two sets based on attribute and threshold.
        lIndices = features(tree.attribute, :) < tree.threshold;
        rIndices = features(tree.attribute, :) >= tree.threshold;
        
        tree.kids{1} = createTree(features(:, lIndices), labels(lIndices));
        tree.kids{2} = createTree(features(:, rIndices), labels(rIndices));
    end
end

%{
%   Input:  Vector containing binary data.
%   Output: Modal value from input vector.
%}

function class = majorityVote(labels)
    
    % Calculating majority like this will handle length(labels) == 0.
    if sum(labels == 0) >= sum(labels == 1)
        class = 0;
    else
        class = 1;
    end
end


% If labels are all the same, will return NaN
function entropy = calcEntropy(y)
    p = sum(y == y(1));
    
    n = length(y) - p;
    entropy = -((p / (p + n)) * log2(p / (p + n))) ...
        - ((n / (n + p)) * log2(n / (n + p)));
end