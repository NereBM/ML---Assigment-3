load('facialPoints.mat');
load('labels.mat');
points = reshape(points, [132, 150]);
%points = transpose(points);

%Mdl = fitcsvm(points,labels,'KernelFunction','rbf','BoxConstraint',1,'KernelScale',0.2);

%[label,score] = predict(Mdl,points(63,:));
label
score

    


%kFold(points,labels,10)

function [results] = kFold(points, labels, k) 

    merged_matrix = cat(1,points,transpose(labels));

    randomised_merged_matrix = merged_matrix(:,randperm(size(merged_matrix,2)));

    randomised_merged_matrix
    points = randomised_merged_matrix(1:132,1:end);
    labels = randomised_merged_matrix(133,1:end);
    
    kFoldsStruct = struct;
    kFoldStruct.test = {};
    kFoldStruct.train = {};
    
    number_of_cols = size(labels,1);
    
    kthFold = number_of_cols/k;
    
    kfold_start = 1;
    kfold_end = kthFold;
    for i = 1:k
        if i == k
            kfold_end = number_of_cols;
        end
        testpoints = points(kfold_star, )
        %%%
        Mdl = fitcsvm(points(kfold_star:kfold_end),labels(,'KernelFunction','rbf','BoxConstraint',1,'KernelScale',0.2);
                
        %%%
        
        kfold_start = kfold_start + kthFold;        
        kfold_end = kfold_end + kthFold;
      
    end
       
    %points
    %labels
    
end
    
    