%This class calculates the f1Score for all the types of Networks

load('treated data/binary smile/binary_test_labels.mat')
load('treated data/binary smile/binary_training_features.mat')
load('treated data/binary smile/binary_test_features.mat')
load('treated data/binary smile/binary_training_labels.mat')

%----------------------------------------------------------------------------------------------
%For DecisionTree

Decision_Tree_A = createTree(binary_training_features, binary_training_labels);
Classified_Predicted_Label_A = classify(binary_test_features, Decision_Tree_A);
f1Score_A = f1Score(Classified_Predicted_Label_A, binary_test_labels);

%-----------------------------------------------------------------------------------------------
%For Artificial Neural Networks Binary Classification

roundNetPredicted = zeros(1,length(binary_test_labels));

net = feedforwardnet(10);
net = train(net, binary_training_features, binary_training_labels);
netPredicted = sim(net, binary_test_features);
%makes sure negative numbers are rounded to zero
for k = 1 : length(netPredicted)
  if(netPredicted(k)<0)
      roundNetPredicted(k) = fix(netPredicted(k));
  else
      roundNetPredicted(k) = round(netPredicted(k));
  end      
end
f1Score_ANN = f1Score(roundNetPredicted, binary_test_labels);
