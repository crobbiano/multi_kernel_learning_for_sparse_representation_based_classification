clear all
clc
close all
%%
load('renko_mean_thrs2_5_len10_.mat')
allClass = label;
allClass(allClass==-1) = 0;
allSet = feature';
allSet = normc(allSet);

%%

fraction = 0.;

classes = unique(allClass);
nObs = length(allClass);
nClasses = length(classes);
nSamples = round(nObs * fraction / nClasses);
nSamples = 500;

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

kfold=200; % Number of classes.
ind=crossvalind('Kfold',allClass,kfold);
% indTrain=logical(sum(ind==1:5,2));
% indTrainSmall=logical(sum(ind==1,2));

indValid=logical(sum(ind==1:15,2));
indValidSmall=logical(sum(ind==6:7,2));
indValidSmall=indValid;

indTest=logical(sum(ind==16:30,2));
indTestSmall=logical(sum(ind==18:19,2));
indTestSmall=indTest;

testClass = allClass(indTest);
testSet=allSet(:,indTest);
testClassSmall=allClass(indTestSmall);
testSetSmall=allSet(:,indTestSmall);

validClass = allClass(indValid);
validSet=allSet(:,indValid);
validSetSmall=allSet(:,indValidSmall);
validClassSmall=allClass(indValidSmall);

%%
save('renko_10_thrs_train.mat', 'trainSet', 'trainClass', 'testSet', 'testClass', 'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall', 'validSet', 'validClass', 'validSetSmall', 'validClassSmall')
