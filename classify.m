%{
%   Input:  Matrix of features, struct containing tree from createTree().
%   Output: Vector containing predicted classes.
%}

function predicted = classify(features, tree)
    
    predicted = zeros(size(features, 2), 1);
    for i = 1: size(features, 2)
        node = tree;
        x = features(:, i);
        
        % Traverse tree until leaf node then set class to leaf node class.
        while ~isempty(node.kids)
            if x(node.attribute) < node.threshold
                node = node.kids{1};
            else
                node = node.kids{2};
            end
        end
        predicted(i) = node.class;
    end
end