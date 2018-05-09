clc
clear
addpath(genpath('srv1_9'));
load('mnist_insitu_save/mnist_insitu_all.mat')
% load('renko.mat')
%%
% Save the dictionary
Dict = dictSetSmall;

% FIXME - subbing in big sets
% testSetSmall = testSet;
% testClassSmall = testClass;
% validSetSmall = validSet;
% validClassSmall = validClass;

% X should be N by M where N is number of atoms and M is number of testing samples. 
X = zeros(size(Dict, 2), size(trainSetSmall, 2));
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
classes = unique(dictClassSmall);
num_classes = numel(classes);
masks = zeros(size(Dict,2),numel(classes));
for i=1:num_classes
    num_samples_per_class(i) = sum(dictClassSmall == classes(i));
    masks(:,i) = dictClassSmall == classes(i);
    locs = find(dictClassSmall == classes(i));
    K_ideal(min(locs):max(locs),min(locs):max(locs)) = 1;
end
textprogressbar(' ');
%% Get ranked ordering of kfncs based on similarity to ideal kernel
textprogressbar('Generating ranks of matrices: ')
for i=1:length(kfncs)
    alignment_scores(i) = kernelAlignment(kernel_mats{i}, K_ideal);
    textprogressbar(i*100/length(kfncs));
end
[sorted, idx] = sort(alignment_scores,'descend');
kernel_mats = kernel_mats(idx);
kfncs = kfncs(idx);
textprogressbar(' ');
%% Compute more kernel matrices
textprogressbar('Generating second set of matrices: ');
for kidx=1:length(kfncs)
    eta_temp = [];
    eta_temp(kidx) = 1; % place a 1 in the current kernel
    
    Hfull{kidx,1}=computeMultiKernelMatrix(Dict,Dict,eta_temp,kfncs);
    for sidx=1:length(trainClassSmall)
        Gfull{kidx,sidx}=computeMultiKernelMatrix(Dict,trainSetSmall(:,sidx),eta_temp,kfncs);
        Bfull{kidx,sidx}=computeMultiKernelMatrix(trainSetSmall(:,sidx),trainSetSmall(:,sidx),eta_temp,kfncs);
    end
    
    textprogressbar(kidx*100/length(kfncs));
end
textprogressbar(' ');
%% Parameters
mu = .02;
% sparsity_reg \lambda
lambda = .1;
% max iterations
T = 10;
% error thresh for convergence
err_thresh = .001;
err = err_thresh + 1;

% Make eta
eta = zeros(length(kfncs),1);
eta(1)=1;

% Find the number of samples in each class
classes = unique(dictClassSmall);
num_classes = numel(classes);
for i=1:num_classes
    num_per_class(i) = sum(dictClassSmall == classes(i));
end
%% Loop to get all sparse coeffs
t = 0;
while(t <= T && err>= err_thresh)
    t = t + 1;
    
    [X, h, g, z(t,:), zm, c, cc_rate_ini, fa_rate_ini] = mklsrcUpdate(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class);
    [C,CM,IND,PER] = confusion(num2bin10(trainClassSmall, max(classes)+1), num2bin10(h, max(classes)+1));
    class_percent_correct_train(:,t) = PER(:,3);
      
    if sum(z(t,:))/length(z(t,:)) == 1 | sum(c)==0
        err = 0;
    else
        [eta, err] = updateEta(eta, c, mu, z(t,:), zm);
    end
    
    display(['TRAINING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(z(t,:))/numel(z(t,:))) ' Error: ' num2str(err)])
end
%% Learn the test set now
t = 0;textprogressbar('TESTING: Generating testing matrices: ')
Hfulltest = Hfull;
for kidx=1:length(kfncs)
    eta_temp = [];
    eta_temp(kidx) = 1; % place a 1 in the current kernel
    
    for sidx=1:length(testClassSmall)
        Gfulltest{kidx,sidx}=computeMultiKernelMatrix(Dict,testSetSmall(:,sidx),eta_temp,kfncs);
        Bfulltest{kidx,sidx}=computeMultiKernelMatrix(testSetSmall(:,sidx),testSetSmall(:,sidx),eta_temp,kfncs);
    end
    textprogressbar(kidx*100/length(kfncs));
end
textprogressbar(' ');

etatest = eta;
err = err_thresh + 1;
t=0; 
while(t <= T && err>= err_thresh)
    t = t + 1;
    [Xtest, htest, gtest, ztest, zmtest, ctest] = mklsrcUpdate(Hfulltest, Gfulltest, Bfulltest, etatest, testClassSmall, classes, num_per_class);
    [C,CM,IND,PER] = confusion(num2bin10(testClassSmall, max(classes)+1), num2bin10(htest, max(classes)+1));
    class_percent_correct_test(:,t) = PER(:,3);
    
    if sum(ztest)/length(ztest) == 1 | sum(ctest)==0
        err = 0;
    else
        [etatest, err] = updateEta(etatest, ctest, mu, ztest, zmtest);
    end
    
    display(['TESTING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(ztest)/numel(ztest)) ' Error: ' num2str(err)])
end
% display(['TESTING: Accuracy: ' num2str(sum(ztest)/numel(ztest))])
%% 'Check OG data again'
display(['TRAINING: 2nd check'])
[Xcheck, hcheck, ~, zcheck, ~, ~, cc_rate_fin, fa_rate_fin] = mklsrcUpdate(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class);
validacc2 = sum(zcheck)/numel(zcheck);
display(['TRAINING: Accuracy: ' num2str(sum(zcheck)/numel(zcheck))])