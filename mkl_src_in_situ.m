clc
clear
addpath(genpath('srv1_9'));
saveSet = 0;
if saveSet
    % load('data/smallRoundBlueCellTumorOFChildhood.mat');
    load('mnist_data.mat');
    Data = double(Data);
    D=Data;
    % clear('Data');
    
    % normalize D
    D=normc(D);
    
    % CV
    kfold=5;
    ind=crossvalind('Kfold',classes,kfold);
    indTest=logical(sum(ind==1:3,2));
    indTrain=logical(sum(ind==4:4,2));
    indNew=logical(sum(ind==5:5,2));
    trainClass=classes(indTrain);
    testClass=classes(indTest);
    
    [~, idxs] = sort(trainClass);
    trainClass = trainClass(idxs);
    trainSet=D(:,indTrain);
    trainSet = trainSet(:,idxs);
    [~, idxs] = sort(testClass);
    testClass = testClass(idxs);
    testSet=D(:,indTest);
    testSet = testSet(:,idxs);
    
    % Get a set of digits 0-3 from the trainSet and 0-4 from the testSet
    trainIdx = logical(sum(trainClass'==0:3, 2));
    testIdx = logical(sum(testClass'==0:4, 2));
    
    savedTrainSet = trainSet(:,trainIdx);
    savedTrainClass = trainClass(trainIdx);
    savedTestSet = testSet(:,testIdx);
    savedTestClass = testClass(testIdx);
    
    
    trainSet = savedTrainSet;
    trainClass = savedTrainClass;
    testSet = savedTestSet;
    testClass = savedTestClass;
    kfold=400;
    ind=crossvalind('Kfold',trainClass,kfold);
    indTrain=logical(sum(ind==1:16,2));
    indValid=logical(sum(ind==17:37,2));
    
    kfold=500;
    ind=crossvalind('Kfold',testClass,kfold);
    indTest=logical(sum(ind==1:20,2));
    
    trainClassSmall=trainClass(indTrain);
    trainSetSmall=trainSet(:,indTrain);
    validClassSmall=trainClass(indValid);
    validSetSmall=trainSet(:,indValid);
    testClassSmall=testClass(indTest);
    testSetSmall=testSet(:,indTest);
    
    save('mnist_train_0-8_test_0-9.mat', 'trainSet', 'trainClass', 'testSet', 'testClass', 'trainSetSmall', 'trainClassSmall', 'testSetSmall', 'testClassSmall', 'validSetSmall', 'validClassSmall')
else
%         load('mnist_train_0-3_test_0-4.mat');
%         load('mnist_train_0-8_test_0-9.mat');
%     load('caltech_35_train.mat');
    load('renko_10_thrs_train.mat');
end
%%
% Save the dictionary
Dict = trainSetSmall;

% FIXME - X should be N by M where N is number of atoms and M is number of
% testing samples. Happens that M == N
X = zeros(size(Dict, 2), size(validSetSmall, 2));
sparsity = zeros(1, size(validSetSmall,2));
ssims = zeros(1, size(validSetSmall,2));
immses = zeros(1, size(validSetSmall,2));
%% Generate kernel fncs
kfncs  = { ...
    @(x,y) x'*y; ...            % Linear
    @(x,y) (x'*y + 1); ...
    @(x,y) (x'*y + 0.5).^2; ...  % Polynomial
    @(x,y) (x'*y + 0.5).^3; ...
    @(x,y) (x'*y + 0.5).^4; ...
    @(x,y) (x'*y + 1.0).^2; ...
    @(x,y) (x'*y + 1.0).^3; ...
    @(x,y) (x'*y + 1.0).^4; ...
    @(x,y) (x'*y + 1.5).^2; ...
    @(x,y) (x'*y + 1.5).^3; ...
    @(x,y) (x'*y + 1.5).^4; ...
    @(x,y) (x'*y + 2.0).^2; ...
    @(x,y) (x'*y + 2.0).^3; ...
    @(x,y) (x'*y + 2.0).^4; ...
    @(x,y) (x'*y + 2.5).^2; ...
    @(x,y) (x'*y + 2.5).^3; ...
    @(x,y) (x'*y + 2.5).^4; ...
    @(x,y) tanh(0.1 + 1.0*(x'*y)); ...  % Hyperbolic Tangent
    @(x,y) tanh(0.2 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.3 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.4 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.5 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.2*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.4*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.6*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.8*(x'*y)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.1)); ...  % Gaussian
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.2)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.3)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.4)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.5)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.6)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.7)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.8)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.9)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.0)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.1)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.2)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.3)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.4)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.5)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.6)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.7)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.8)); ...
    };
%% Compute the M kernel matrices
textprogressbar('Generating first set of matrices: ')
% Find K_m(Y, Y) for all M kernel functions
kernel_mats = cell(length(kfncs), 1);
for m=1:length(kfncs)
    option.kernel = 'cust'; option.kernelfnc=kfncs{m};
    kernel_mats{m} = computeKernelMatrix(Dict,Dict,option);
    textprogressbar(m*100/length(kfncs));
end

% Make the ideal matrix - FIXME - assumes blocks of samples (probably fine)
K_ideal = eye(size(Dict,2));
% Find the number of samples in each class
classes = unique(trainClassSmall);
num_classes = numel(classes);
masks = zeros(size(Dict,2),numel(classes));
for i=1:num_classes
    num_samples_per_class(i) = sum(trainClassSmall == classes(i));
    masks(:,i) = trainClassSmall == classes(i);
    locs = find(trainClassSmall == classes(i));
    K_ideal(min(locs):max(locs),min(locs):max(locs)) = 1;
end
textprogressbar(' Done.');
%% Get ranked ordering of kfncs based on similarity to ideal kernel
textprogressbar('Generating ranks of matrices: ')
for i=1:length(kfncs)
    alignment_scores(i) = kernelAlignment(kernel_mats{i}, K_ideal);
    textprogressbar(i*100/length(kfncs));
end
[sorted, idx] = sort(alignment_scores,'descend');
kernel_mats = kernel_mats(idx);
kfncs = kfncs(idx);
textprogressbar(' Done.');
%% Compute more kernel matrices
% if iscell(Hfull) == 0
    textprogressbar('Generating second set of matrices: ');
    for kidx=1:length(kfncs)
        eta_temp = [];
        eta_temp(kidx) = 1; % place a 1 in the current kernel
        
        Hfull{kidx,1}=computeMultiKernelMatrix(Dict,Dict,eta_temp,kfncs);
        for sidx=1:length(validClassSmall)
            Gfull{kidx,sidx}=computeMultiKernelMatrix(Dict,validSetSmall(:,sidx),eta_temp,kfncs);
            Bfull{kidx,sidx}=computeMultiKernelMatrix(validSetSmall(:,sidx),validSetSmall(:,sidx),eta_temp,kfncs);
        end
        
        textprogressbar(kidx*100/length(kfncs));
    end
    textprogressbar(' Done.');
% end
%% Generate eta
eta = zeros(length(kfncs),1);
eta(1)=1;
%% Parameters
mu = .002;
% sparsity_reg \lambda
lambda = .1;
% max iterations
T = 20;
% error thresh for convergence
err_thresh = .001;
err = err_thresh + 1;

optionKSRSC.lambda=lambda;
% optionKSRSC.SCMethod='l1qpAS'; % can be nnqpAS, l1qpAS, nnqpIP, l1qpIP, l1qpPX, nnqpSMO, l1qpSMO
optionKSRSC.iter=200;
optionKSRSC.dis=0;
optionKSRSC.residual=1e-4;
optionKSRSC.tof=1e-4;
%% Loop to get all sparse coeffs
% Find the number of samples in each class
classes = unique(trainClassSmall);
num_classes = numel(classes);
for i=1:num_classes
    num_per_class(i) = sum(trainClassSmall == classes(i));
end

t = 0;
while(t <= T && err>= err_thresh)
    t = t + 1;
    
    H=computeMultiKernelMatrixFromPrecomputed(Hfull,eta);
    
    textprogressbar('Training Coeff progress: ');
    for idx = 1:size(validSetSmall, 2)
        testSet = validSetSmall(:,idx);
        testClass = validClassSmall(idx);
        
        % compute kernels
        G=computeMultiKernelMatrixFromPrecomputed(Gfull(:, idx),eta);
        B=computeMultiKernelMatrixFromPrecomputed(Bfull(:, idx),eta);
        
        % KSRSC sparse coding
%         [X(:, idx), ~, sparsity(idx)] =KSRSC(H,G,diag(B),optionKSRSC);
        Xtemp(:,idx) = OMP(H, G, 50, 0);
        X(:, idx) = Xtemp(:, idx);
%         sparsity(idx) = (numel(X(:,idx)) - sum(X(:,idx)>0) )/ numel(X(:,idx));
%         ssims(idx) = ssim(reshape(Dict*X(:,idx), 28, 28), reshape(validSetSmall(:,idx), 28, 28));
        immses(idx) = immse(Dict*X(:,idx), validSetSmall(:,idx));
        
        % Find class - calculate h (class) and z (correct class)
        classerr = zeros(1, num_classes);
        for class=1:num_classes
            b_idx = sum(num_per_class(1:class-1)) + 1;
            e_idx = sum(num_per_class(1:class));
            x_c = X(b_idx:e_idx ,idx);
            kernel_c = H(b_idx:e_idx ,b_idx:e_idx);
            partial_c = G(b_idx:e_idx)';
            classerr(class) = B + x_c'*kernel_c*x_c - 2*partial_c*x_c;
        end
        [~, h(idx)] = min(classerr);
        h(idx) = h(idx) - 1;  % Adjust for indexing in matlab
        z(idx) = (h(idx) == validClassSmall(idx));
        
        % Need to calculate the ability to classify for each individual kernel
        for kidx=1:length(kfncs)
            eta_temp = [];
            eta_temp(kidx) = 1; % place a 1 in the current kernel
            
            H_temp = computeMultiKernelMatrixFromPrecomputed(Hfull, eta_temp);
            G_temp = computeMultiKernelMatrixFromPrecomputed(Gfull(:, idx),eta_temp);
            B_temp = computeMultiKernelMatrixFromPrecomputed(Bfull(:, idx),eta_temp);
            
            err_temp = zeros(1, num_classes);
            for class=1:num_classes
                b_idx = sum(num_per_class(1:class-1)) + 1;
                e_idx = sum(num_per_class(1:class));
                x_c = X(b_idx:e_idx ,idx);
                kernel_c = H_temp(b_idx:e_idx ,b_idx:e_idx);
                partial_c = G_temp(b_idx:e_idx)';
                err_temp(class) = B_temp + x_c'*kernel_c*x_c - 2*partial_c*x_c;
            end
            [~, h_temp] = min(err_temp);
            h_temp = h_temp - 1;  % Adjust for indexing in matlab
            g(kidx, idx) = h_temp;
            zm(kidx, idx) = g(kidx, idx) == validClassSmall(idx);
            
            if sum(1-z)
                c(kidx,1) = sum(zm(kidx, find(z==0)))/sum(1-z);
            else
                c(kidx,1) = 0;
            end
        end
        
        textprogressbar(idx*100/size(validSetSmall, 2));
    end
    
    textprogressbar(' Done.');
    
    
    % Calculate the C values
    if sum(z)/length(z) == 1 | sum(c)==0
        err = 0;
    else
        [best_c_val,best_c_idx] = max(c); % THIS WORKS
        best_c_val_og = best_c_val;
        update_c = 1;
        if update_c
            for i=best_c_idx:-1:1
                if (c(i) ~=0) & (c(i) + mu > best_c_val_og)  % THIS WORKS
                    best_c_idx = i;
                    best_c_val = c(i);
                    display(['Changed best_c to higher index: ' num2str(i)])
                end
            end
        end
        new_kernel = best_c_idx;
        new_kernel_weight = sum(bitand(zm(new_kernel,:), not(z)))/sum(bitor(not(z), not(zm(new_kernel,:))));
        curr_kernel_weight = sum(bitand(z, not(zm(new_kernel,:))))/sum(bitor(not(z), not(zm(new_kernel,:))));
        prev_eta = eta; % save for calcing error
        % Do the actual mixing weight update here
        for m=1:length(kfncs)
            if m==new_kernel
                eta(m)=new_kernel_weight;
            else
                eta(m) = eta(m)*curr_kernel_weight;
            end
        end
        
        % Compute sums of all weights and normalize weights by sum
        total_weights = sum(eta);
        eta = eta/total_weights;
        
        % set err = ||eta^{t-1}-eta^{t}||_2
        err = norm(prev_eta - eta,2);
    end
    display(['Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(z)/numel(z)) ' Error: ' num2str(err)])
end
%% Look at the recon errors
errors = validSetSmall - Dict*X;
errornorms = vecnorm(errors);
%% Test the test set
textprogressbar('Generating testing matrices: ')
for kidx=1:length(kfncs)
    eta_temp = [];
    eta_temp(kidx) = 1; % place a 1 in the current kernel
    
    Hfulltest{kidx,1}=computeMultiKernelMatrix(Dict,Dict,eta_temp,kfncs);
    for sidx=1:length(testClassSmall)
        Gfulltest{kidx,sidx}=computeMultiKernelMatrix(Dict,testSetSmall(:,sidx),eta_temp,kfncs);
        Bfulltest{kidx,sidx}=computeMultiKernelMatrix(testSetSmall(:,sidx),testSetSmall(:,sidx),eta_temp,kfncs);
    end
    textprogressbar(kidx*100/length(kfncs));
end
textprogressbar(' Done.');
textprogressbar('Testing prediction progress: ');
Htest=computeMultiKernelMatrixFromPrecomputed(Hfulltest,eta);
for idx = 1:size(testSetSmall, 2)
    testSet = testSetSmall(:,idx);
    testClass = testClassSmall(idx);
    
    % compute kernels
    Gtest=computeMultiKernelMatrixFromPrecomputed(Gfulltest(:, idx),eta);
    Btest=computeMultiKernelMatrixFromPrecomputed(Bfulltest(:, idx),eta);
    
    % KSRSC sparse coding
%     [Xtest(:, idx), ~, sparsitytest(idx)] =KSRSC(Htest,Gtest,diag(Btest),optionKSRSC);
    Xtemptest(:,idx) = OMP(Htest, Gtest, 50, 0.001);
    Xtest(:,idx) = Xtemptest(:,idx);
%     sparsitytest(idx) = (numel(Xtest(:,idx)) - sum(Xtest(:,idx)>0) )/ numel(Xtest(:,idx));
%     ssimstest(idx) = ssim(reshape(Dict*Xtemptest(:,idx), 28, 28), reshape(testSetSmall(:,idx), 28, 28));
    immsestest(idx) = immse(Dict*Xtemptest(:,idx), testSetSmall(:,idx));
    
    % Find class - calculate h (class) and z (correct class)
    classerr = zeros(1, num_classes);
    for class=1:num_classes
        b_idx = sum(num_per_class(1:class-1)) + 1;
        e_idx = sum(num_per_class(1:class));
        x_c = Xtest(b_idx:e_idx ,idx);
        kernel_c = Htest(b_idx:e_idx ,b_idx:e_idx);
        partial_c = Gtest(b_idx:e_idx)';
        classerr(class) = Btest + x_c'*kernel_c*x_c - 2*partial_c*x_c;
    end
    [~, htest(idx)] = min(classerr);
    htest(idx) = htest(idx) - 1;  % Adjust for indexing in matlab
    ztest(idx) = (htest(idx) == testClassSmall(idx));
    
    textprogressbar(idx*100/size(testSetSmall, 2));
end
textprogressbar(' Done.');

display(['Testing Accuracy: ' num2str(sum(ztest)/numel(ztest))])

%% Get just the top K largest coeffs in the Xtest for analysis
Xtestsparse = Xtemptest;
% K=3;
% totalnums = 1:length(Xtestsparse(:,1));
% for i=1:size(Xtestsparse,2)
%     [~, maxes]=maxk(Xtestsparse(:,i), K);
%     Xtestsparse(setdiff(totalnums, maxes),i) = 0;
% end
recons = Dict*Xtestsparse;
recon_errors = testSetSmall - recons;
errors = vecnorm(recon_errors);
