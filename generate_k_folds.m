function [k_folds] = generate_k_folds(n,k)
    % Takes as input the number of items needing to be partitioned, as well
    % as k, the number of folds. It will return a struct containing in 
    %.test indices of each of the k folds. And in .train the rest of the
    %indices that aren't to be trained
    
    kFoldsStruct = struct;
    kFoldStruct.test = {k};
    kFoldStruct.train = {k};
    
    random_indices = randperm(n);
    
    kthFold = floor(n/k);
    
    kfold_start = 1;
    kfold_end = kthFold;
    for i = 1:k
        if i == k
            kfold_end = n;
        end
        %%%
        kFoldStruct.test{i} = random_indices(kfold_start:kfold_end);    
        kFoldStruct.train{i} = setdiff(random_indices,  random_indices(kfold_start:kfold_end));       
        %%%
        
        kfold_start = kfold_start + kthFold;        
        kfold_end = kfold_end + kthFold;
      
    end

    k_folds = kFoldStruct;
    

end
