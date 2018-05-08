clear all
clc
close all
%%
load('renko_min_th2_len20.mat')
allClass = label;
allClass(allClass==-1) = 0;
allSet = feature';
allSet = normc(allSet);
%%
classes = unique(allClass);
nObs = length(allClass);
nClasses = length(classes);
nSamples = 200;

for ii = 1:nClasses
    idx = allClass == classes(ii);
    idxs((ii-1)*nSamples+1:ii*nSamples) = randsample(allClass(idx), nSamples);
    
    sampleidx = find(allClass == classes(ii));
    indDict((ii-1)*nSamples+1:ii*nSamples) = randsample(sampleidx, nSamples);
end

numtotalvec = 1:length(allClass);
remainingindices = setdiff(numtotalvec, indDict);
indDictSmall = indDict;

dictClass = allClass(indDict);
dictSet=allSet(:,indDict);
dictClassSmall=allClass(indDictSmall);
dictSetSmall=allSet(:,indDictSmall);

allClass = allClass(remainingindices);
allSet = allSet(:,remainingindices);
%%

kfold=140; % Number of classes.
ind=crossvalind('Kfold',allClass,kfold);
indTrain=logical(sum(ind==31:41,2));
indTrainSmall=logical(sum(ind==31:33,2));
% indTrainSmall=indTrain;

indValid=logical(sum(ind==1:15,2));
indValidSmall=logical(sum(ind==6:7,2));
% indValidSmall=indValid;

indTest=logical(sum(ind==16:30,2));
indTestSmall=logical(sum(ind==18:19,2));
% indTestSmall=indTest;

trainClass = allClass(indTrain);
trainSet=allSet(:,indTrain);
trainClassSmall=allClass(indTrainSmall);
trainSetSmall=allSet(:,indTrainSmall);

testClass = allClass(indTest);
testSet=allSet(:,indTest);
testClassSmall=allClass(indTestSmall);
testSetSmall=allSet(:,indTestSmall);

validClass = allClass(indValid);
validSet=allSet(:,indValid);
validSetSmall=allSet(:,indValidSmall);
validClassSmall=allClass(indValidSmall);

%% Get only certain digits
someDigits = 0;
classes = [0 2 3 8]
if someDigits
    dictIdx       = logical(sum(dictClass'       == classes, 2));
    dictIdxSmall  = logical(sum(dictClassSmall'  == classes, 2));
    trainIdx      = logical(sum(trainClass'      == classes, 2));
    trainIdxSmall = logical(sum(trainClassSmall' == classes, 2));
    testIdx       = logical(sum(testClass'       == classes, 2));
    testIdxSmall  = logical(sum(testClassSmall'  == classes, 2));
    validIdx      = logical(sum(validClass'      == classes, 2));
    validIdxSmall = logical(sum(validClassSmall' == classes, 2));
    
    dictClassSmall = dictClass(dictIdxSmall);
    dictSetSmall   = dictSet(:,dictIdxSmall);
    dictClass      = dictClass(dictIdx);
    dictSet        = dictSet(:,dictIdx);
    
    trainClass      = trainClass(trainIdx);
    trainSet        = trainSet(:,trainIdx);
    trainClassSmall = trainClassSmall(trainIdxSmall);
    trainSetSmall   = trainSetSmall(:,trainIdxSmall);
    
    testClass      = testClass(testIdx);
    testSet        = testSet(:,testIdx);
    testClassSmall = testClassSmall(testIdxSmall);
    testSetSmall   = testSetSmall(:,testIdxSmall);
    
    validClass      = validClass(validIdx);
    validSet        = validSet(:,validIdx);
    validSetSmall   = validSetSmall(:,validIdxSmall);
    validClassSmall = validClassSmall(validIdxSmall);
end
%%
save('renko.mat', 'dictClass', 'dictClassSmall', 'dictSet', ...
    'dictSetSmall', 'trainSet', 'trainClass', 'testSet', 'testClass',...
    'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall',...
    'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
