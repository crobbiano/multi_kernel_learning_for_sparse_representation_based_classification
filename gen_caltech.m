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
nSamples = 31;

for ii = 1:nClasses
    idx = allClass == classes(ii);
    idxs((ii-1)*nSamples+1:ii*nSamples) = randsample(allClass(idx), nSamples);
    
    
    sampleidx = find(allClass == classes(ii));
    indTrain((ii-1)*nSamples+1:ii*nSamples) = randsample(sampleidx, nSamples);
end

numtotalvec = 1:length(allClass);
remainingindices = setdiff(numtotalvec, indTrain);
indTrainSmall = indTrain;

trainClass = allClass(indTrain);
trainSet=allSet(:,indTrain);
trainClassSmall=allClass(indTrainSmall);
trainSetSmall=allSet(:,indTrainSmall);

allClass = allClass(remainingindices);
allSet = allSet(:,remainingindices);
%%

kfold=30; % Number of classes.
ind=crossvalind('Kfold',allClass,kfold);
% indTrain=logical(sum(ind==1:5,2));
% indTrainSmall=logical(sum(ind==1,2));

indValid=logical(sum(ind==1:15,2));
indValidSmall=logical(sum(ind==6:7,2));

indTest=logical(sum(ind==16:30,2));
indTestSmall=logical(sum(ind==18:19,2));

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
save('caltech_35_train.mat', 'trainSet', 'trainClass', 'testSet', 'testClass', 'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall', 'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
