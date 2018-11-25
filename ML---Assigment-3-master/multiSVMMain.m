clear
load('multiclassEmotion/emotions_data.mat');
multiPoints = x;
multiLabels = y';

%{
t = templateSVM('Standardize',1,'KernelFunction','Polynomial' ...
    , 'BoxConstraint',1);

fitcecoc( multiPoints, multiLabels                          ...
         , 'Learners',t );
%}

classParam = struct;
classParam.c = [0.1, 1, 10, 100, 500 1000];
classParam.kernel = 'rbf';
classParam.paramString = 'KernelScale';
classParam.kernelParam = [0.1, 1, 10, 100, 500];
rbfSVMData = trainSVMClass(multiPoints, multiLabels, 10, classParam);

%{
classParam = struct;
classParam.c = [0.01, 0.1, 1, 10, 100, 500];
classParam.kernel = "polynomial";
classParam.paramString = "PolynomialOrder";
classParam.kernelParam = [2 3 4];
polySVMdata = binarySVM(binaryPoints, binaryLabels, 10, classParam);
%}
