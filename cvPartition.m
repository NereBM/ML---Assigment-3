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

function partitions = cvPartition(N, k)

    partitions       = struct;
    partitions.train = {k};
    partitions.test  = {k};
    partitionSize    = round(N / k);
    
    for i = 1:k
        testPartition = [];       
        while length(testPartition) < partitionSize
            randN = randi(N);            
            % Check randN not in testPartition and not in partitions.test
            if ~any(testPartition == randN) ...
                & ~cellfun(@(x) any(x == randN), partitions.test)
                testPartition = [testPartition randN];
            end
        end
        partitions.test{i} = testPartition;
        partitions.train{i} = setdiff(1:N, testPartition);
    end

end