clear all
clc
close all
%%
load('spatialpyramidfeatures4caltech101/spatialpyramidfeatures4caltech101.mat')
allClass = bin102num(labelMat);
allSet = featureMat;

%%

fraction = 0.;

classes = unique(allClass);
nObs = length(allClass);
nClasses = length(classes);
nSamples = round(nObs * fraction / nClasses);
nSamples = 10;

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

kfold=30; % Number of classes.
ind=crossvalind('Kfold',allClass,kfold);
indTrain=logical(sum(ind==1:5,2));
indTrainSmall=logical(sum(ind==1:2,2));

indValid=logical(sum(ind==6:15,2));
indValidSmall=logical(sum(ind==6:7,2));

indTest=logical(sum(ind==16:30,2));
indTestSmall=logical(sum(ind==18:19,2));

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

%% Pick just 5 classes
pick5 = 1;
if pick5
    classes = [1 2 3 4 5];
    
    indDict = find(sum(dictClass'==classes,2));
    indDictSmall = indDict;
    dictClassSmall = dictClass(indDictSmall);
    dictClass = dictClass(indDict);
    dictSetSmall = dictSet(:,indDictSmall);
    dictSet = dictSet(:,indDict);
    
    indTrain = find(sum(trainClass'==classes,2));
    indTrainSmall = indTrain;
    trainClassSmall = trainClass(indTrainSmall);
    trainClass = trainClass(indTrain);
    trainSetSmall = trainSet(:,indTrainSmall);
    trainSet = trainSet(:,indTrain);

    indTest = find(sum(testClass'==classes,2));
    indTestSmall = indTest;
    testClassSmall = testClass(indTestSmall);
    testClass = testClass(indTest);
    testSetSmall = testSet(:,indTestSmall);
    testSet = testSet(:,indTest);

    indValid = find(sum(validClass'==classes,2));
    indValidSmall = indValid;
    validClassSmall = validClass(indValidSmall);
    validClass = validClass(indValid);
    validSetSmall = validSet(:,indValidSmall);
    validSet = validSet(:,indValid);

end

%%
save('caltech_15_train.mat', 'dictSet','dictSetSmall', 'dictClass', 'dictClassSmall', 'trainSet', 'trainClass', 'testSet', 'testClass', 'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall', 'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
