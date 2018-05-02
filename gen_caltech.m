clear all
clc
close all
%%

useCaltech = 1;
if useCaltech
    imgsz = 144;
    
    imds = imageDatastore('101_ObjectCategories', 'IncludeSubfolders',true,'LabelSource','foldernames');
    numTrainFiles = 15;
    [imdsTrain,imdsTest] = splitEachLabel(imds,numTrainFiles,'randomize');
    
    trainClasstxt = imdsTrain.Labels;
    trainClass = grp2idx(trainClasstxt);
    trainClass = trainClass';
    for i=1:length(imdsTrain.Files)
        im = imread(imdsTrain.Files{i});
        if size(im, 3) > 1
            im = rgb2gray(im);
        end
        %         figure(1); imshow(im,[])
        im = imresize(im, [imgsz imgsz]);
        %         figure(2); imshow(double(im),[])
        trainSet(:,i) = reshape(double(im), imgsz*imgsz, 1);
        %         figure(3); imshow(reshape(Y(:,i), 128, 128),[])
    end
    
    testClasstxt = imdsTest.Labels;
    testClass = grp2idx(testClasstxt);
    testClass = testClass';
    for i=1:length(imdsTest.Files)
        im = imread(imdsTest.Files{i});
        if size(im, 3) > 1
            im = rgb2gray(im);
        end
        im = imresize(im, [imgsz imgsz]);
        testSet(:,i) = reshape(double(im), imgsz*imgsz, 1);
    end
    
    [~, idxs] = sort(trainClass);
    trainClass = trainClass(idxs);
    trainSet = trainSet(:,idxs);
    trainSet = normc(trainSet);
    [~, idxs] = sort(testClass);
    testClass = testClass(idxs);
    testSet = testSet(:,idxs);
    testSet = normc(testSet);
    
    kfold=5;
    ind=crossvalind('Kfold',trainClass,kfold);
    indSmall=logical(sum(ind==1:3,2));
    trainClassSmall=trainClass(indSmall);
    trainSetSmall=trainSet(:,indSmall);
    
    kfold=20;
    ind=crossvalind('Kfold',testClass,kfold);
    indSmall=logical(sum(ind==1:3,2));
    indValid=logical(sum(ind==4:5,2));
    testClassSmall=testClass(indSmall);
    testSetSmall=testSet(:,indSmall);
    validClassSmall=testClass(indValid);
    validSetSmall=testSet(:,indValid);
    
end

%%
save('caltech_train.mat', 'trainSet', 'trainClass', 'testSet', 'testClass', 'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall', 'validSetSmall', 'validClassSmall')
