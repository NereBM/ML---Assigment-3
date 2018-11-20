%{
%   Input: N: Total number of elements to partition
%          k: Number of partitions to create.
%
%   Output: Struct with following format:
%
%   struct partitions {
%       train - cell array containing k vectors of indices for training.
%       test - cell array containing k vectors of indices for testing.
%   }
%}

function partitions = partition(N, k)

    partitions       = struct;
    partitions.train = {k};
    partitions.test  = {k};
    partitionSize    = round(N / k);
    
    % 1 indicates value used, 0 not used. Allows O(1) lookup.
    numsUsed = zeros(1, N);
    
    for i = 1:k
        testPartition = [];       
        while length(testPartition) < partitionSize
            randN = randi(N);
            
            % Check randN not used previously, look up index in numsUsed.
            if ~numsUsed(randN)
                testPartition = [testPartition randN];
                numsUsed(randN) = 1;
            end
        end
        partitions.test{i} = testPartition;
        partitions.train{i} = setdiff(1:N, testPartition);
    end
end