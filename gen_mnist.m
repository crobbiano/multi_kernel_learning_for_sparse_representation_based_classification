clear all
clc
close all
%%
load('mnist_data.mat')
allClass = classes;
allSet = Data;
allSet = normc(allSet);

%%
classes = unique(allClass);
nObs = length(allClass);
nClasses = length(classes);
nSamples = 25;

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
indTrainSmall=logical(sum(ind==1,2));
indTrainSmall=indTrain;

indValid=logical(sum(ind==1:15,2));
indValidSmall=logical(sum(ind==6:7,2));
indValidSmall=indValid;

indTest=logical(sum(ind==16:30,2));
indTestSmall=logical(sum(ind==18:19,2));
indTestSmall=indTest;

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
someDigits = 1;
if someDigits
    dictIdx       = logical(sum(dictClass'       == 0:3, 2));
    dictIdxSmall  = logical(sum(dictClassSmall'  == 0:3, 2));
    trainIdx      = logical(sum(trainClass'      == 0:3, 2));
    trainIdxSmall = logical(sum(trainClassSmall' == 0:3, 2));
    testIdx       = logical(sum(testClass'       == 0:4, 2));
    testIdxSmall  = logical(sum(testClassSmall'  == 0:4, 2));
    validIdx      = logical(sum(validClass'      == 0:4, 2));
    validIdxSmall = logical(sum(validClassSmall' == 0:4, 2));
    
    dictClass      = dictClass(dictIdx);
    dictSet        = dictSet(:,dictIdx);
    dictClassSmall = dictClass(dictIdxSmall);
    dictSetSmall   = dictSet(:,dictIdxSmall);
    
    trainClass      = allClass(trainIdx);
    trainSet        = allSet(:,trainIdx);
    trainClassSmall = allClass(trainIdxSmall);
    trainSetSmall   = allSet(:,trainIdxSmall);
    
    testClass      = allClass(testIdx);
    testSet        = allSet(:,testIdx);
    testClassSmall = allClass(testIdxSmall);
    testSetSmall   = allSet(:,testIdxSmall);
    
    validClass      = allClass(validIdx);
    validSet        = allSet(:,validIdx);
    validSetSmall   = allSet(:,validIdxSmall);
    validClassSmall = allClass(validIdxSmall);
end

%%
save('mnist_insitu_0-3_0-4.mat', 'dictClass', 'dictClassSmall', 'dictSet', ...
    'dictSetSmall', 'trainSet', 'trainClass', 'testSet', 'testClass',...
    'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall',...
    'validSet', 'validClass', 'validSetSmall', 'validClassSmall')