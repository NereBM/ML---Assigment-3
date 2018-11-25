%{
%   Input:  Matrix of features, vector of corresponding labels.
%
%   Output: The best attribute, as measured using the ID3 algorithm as
%            well as the corresponding threshold which splits the data.
%
%}

function [best_feature,best_threshold] = choose_attribute(features,targets)
    targets_copy = transpose(targets);
        
    p = sum(targets_copy==1);   % Stores total number of ps
    
    n = sum(targets_copy==0);   % Stores total number of ns
    
    number_of_rows = size(features,1);
    number_of_cols = size(features,2);
    
    max_gain = 0;
    max_threshold = 0;
        
    for i=1:number_of_rows % loops over rows
        targets_copy = transpose(targets);
        current_feature = features(i,:);
        
        %%%% This section of code is to sort the feature and labels %%%
        current_labels = targets_copy;
        current_merged = cat(1,current_feature,current_labels); % merge label and feature
        
        [temp, order] = sort(current_merged(1,:));  % sort based on feature values
        sorted_merged = current_merged(:,order);
        
        current_feature = sorted_merged(1,:);   % unpack the feature now sorted
        targets_copy = sorted_merged(2,:);   % unpack the label
        
        
        for j=1:(number_of_cols-1)  % Partitions the set to the left and right of the threshold
            p1 = sum(targets_copy(1:j) == 1);
            n1 = sum(targets_copy(1:j) == 0);
                        
            p2 = sum(targets_copy(j+1:end) == 1);
            n2 = sum(targets_copy(j+1:end) == 0);
               
            % Calculate the remainder
            remainder = (((n1+p1)/(n+p))*generate_I(n1,p1)) + (((n2+p2)/(n+p))*generate_I(n2,p2));
            
            % Calculate the gain
            gain = generate_I(n,p) - remainder;
            
            if gain > max_gain  % Checks if the calculated gain is greater than the maximum gain, if so, updates it
                max_gain = gain;
                max_column_index = j;
                max_threshold = (current_feature(j)+current_feature(j+1))/2;
                feature = i;              
                
            end                                    
        end
    end

    gain = max_gain;
    best_feature = feature;
    threshold_index = max_column_index;
    best_threshold = max_threshold;  % numerical threshold
   
end

function [value] = generate_I(n,p)
    % Calculates the entropy I for a given n and p.
    value = ((-p/(p+n)) * log2(p/(p+n))) - ((n/(n+p)) * log2(n/(n+p)));
    if isnan(value) % Instead of just returning NaN which results in being unable to calculate gain, this sets anything NaN to 0
        value = 0;
    end
end
