clc
clear
addpath(genpath('srv1_9'));
load('mnist_insitu_save/mnist_insitu_0_3_6_8.mat')
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

% First, evaluate the validation set
textprogressbar('GENERALIZATION: Generating validation matrices: ')
for kidx=1:length(kfncs)
    eta_temp = [];
    eta_temp(kidx) = 1; % place a 1 in the current kernel
    
    Hfullvalid{kidx,1}=computeMultiKernelMatrix(Dict,Dict,eta_temp,kfncs);
    for sidx=1:length(validClassSmall)
        Gfullvalid{kidx,sidx}=computeMultiKernelMatrix(Dict,validSetSmall(:,sidx),eta_temp,kfncs);
        Bfullvalid{kidx,sidx}=computeMultiKernelMatrix(validSetSmall(:,sidx),validSetSmall(:,sidx),eta_temp,kfncs);
    end
    textprogressbar(kidx*100/length(kfncs));
end
textprogressbar(' ');
%% Parameters
mu = .02;
% sparsity_reg \lambda
lambda = .1;
% max iterations
T = 6;
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
    
    [Xvalid, hvalid, ~, zvalid(t,:), ~, ~] = mklsrcUpdate(Hfullvalid, Gfullvalid, Bfullvalid, eta, validClassSmall, classes, num_per_class);
    validacc1 = sum(zvalid(t,:))/numel(zvalid(t,:));
    display(['GENERALIZATION: Accuracy: ' num2str(sum(zvalid(t,:))/numel(zvalid(t,:)))])

    
    if sum(z(t,:))/length(z(t,:)) == 1 | sum(c)==0
        err = 0;
    else
        [eta, err] = updateEta(eta, c, mu, z(t,:), zm);
    end
    
    display(['TRAINING: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(z(t,:))/numel(z(t,:))) ' Error: ' num2str(err)])
end

%
plotTrain = 1;
if plotTrain == 1
    figure(98); clf; hold on
    idx = 1;
    clear c b
    for i=1:size(class_percent_correct_train, 1)
        if sum(class_percent_correct_train(i, :)) > 0
            color = [mod((3*i/15),1) mod((7*(i+1)/15),1) 1-(i/10)];
            b(idx) = plot(class_percent_correct_train(i, :),'--', 'Color', color);
            text(1:length(class_percent_correct_train(i, :)), class_percent_correct_train(i, :), num2str(i-1))
            cc{idx} = [num2str(i-1) ' - Train'];
            idx = idx + 1;
        end
    end
%     xlim([1 t])
    xlabel('Baseline Learning Iteration'); ylabel('Correct Classification Rate')
    legend(b,cc)
    disp(['Post-Baseline Learning CC Rate (Training Contacts): ' num2str(mean(z(end,:)))]) 
end
%% Learn the test set now
t = 0;textprogressbar('INSITU: Generating testing matrices: ')
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

% etatest = zeros(size(eta));
etatest = eta;
% etatest = zeros(size(eta));
% etatest(1)=1;
err = err_thresh + 1;
t=0; 
while(t <= T && err>= err_thresh)
    t = t + 1;
    [Xtest, htest, gtest, ztest, zmtest, ctest] = mklsrcUpdate(Hfulltest, Gfulltest, Bfulltest, etatest, testClassSmall, classes, num_per_class);
    [C,CM,IND,PER] = confusion(num2bin10(testClassSmall, max(classes)+1), num2bin10(htest, max(classes)+1));
    class_percent_correct_test(:,t) = PER(:,3);
    
    % more baseline stuff
    [X, h, g, z(end+1,:), zm, c] = mklsrcUpdate(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class);
    [C,CM,IND,PER] = confusion(num2bin10(trainClassSmall, max(classes)+1), num2bin10(h, max(classes)+1));
    class_percent_correct_train(:,end+1) = PER(:,3);
    
    [Xvalid, hvalid, ~, zvalid(end+1,:), ~, ~] = mklsrcUpdate(Hfullvalid, Gfullvalid, Bfullvalid, etatest, validClassSmall, classes, num_per_class);
    
    if sum(ztest)/length(ztest) == 1 | sum(ctest)==0
        err = 0;
    else
        [etatest, err] = updateEta(etatest, ctest, mu, ztest, zmtest);
    end
    
    display(['INSITU: Iteration: ' num2str(t) '/' num2str(T) ' Accuracy: ' num2str(sum(ztest)/numel(ztest)) ' Error: ' num2str(err)])
end

display(['INSITU: Accuracy: ' num2str(sum(ztest)/numel(ztest))])
%
plotTest = 1;
if plotTest== 1
    figure(97); clf; hold on
    idx = 1;
    clear c b
    for i=1:size(class_percent_correct_test, 1)
        if sum(class_percent_correct_test(i, :)) > 0
            color = [mod((3*i/15),1) mod((7*(i+1)/15),1) 1-(i/10)];
            b(idx) = plot(class_percent_correct_test(i, :),'--', 'Color', color);
            text(1:length(class_percent_correct_test(i, :)), class_percent_correct_test(i, :), num2str(i-1))
            cc{idx} = [num2str(i-1) ' - Test'];
            idx = idx + 1;
%             b(idx) = plot(results.correct_class_percent_gen(i, :), 'Color', color);
%             c{idx} = [num2str(i-1) ' - Gen'];
%             idx = idx + 1;
        end
    end
%     xlim([1 t])
    xlabel('In-Situ Learning Iteration'); ylabel('Correct Classification Rate')
    legend(b,cc)
    disp(['In-Situ Learning CC Rate (Generalization Contacts): ' num2str(mean(ztest))]) 
    
    % More baseline
    figure(96); clf; hold on
    idx = 1;
    clear c b
    for i=1:size(class_percent_correct_train, 1)
        if sum(class_percent_correct_train(i, :)) > 0
            color = [mod((3*i/15),1) mod((7*(i+1)/15),1) 1-(i/10)];        
            subplot(2,5,i)
            b(idx) = plot(class_percent_correct_train(i, :),'--', 'Color', color);
            text(1:length(class_percent_correct_train(i, :)), class_percent_correct_train(i, :), num2str(i-1))
            cc{idx} = [num2str(i-1) ' - Test'];
            idx = idx + 1;
            grid minor
            ylim([.7 1])
            ylabel([num2str(i-1) ' - Correct Classification Rate'])
            xlabel('In-Situ Learning Iteration');
            %             b(idx) = plot(results.correct_class_percent_gen(i, :), 'Color', color);
            %             c{idx} = [num2str(i-1) ' - Gen'];
            %             idx = idx + 1;
        end
    end
    %     xlim([1 t])
    xlabel('Total Learning Iteration'); ylabel('Correct Classification Rate')
%     legend(b,cc)
    disp(['Post-In-Situ Learning CC Rate (Training Contacts): ' num2str(mean(z(end,:)))])

end
%% Validate against generalization set
display(['GENERALIZATION: 2nd check'])
% [Xvalid, hvalid, ~, zvalid, ~, ~] = mklsrcUpdate(Hfullvalid, Gfullvalid, Bfullvalid, etatest, validClassSmall, classes, num_per_class);
% 
% validacc2 = sum(zvalid)/numel(zvalid);
% display(['GENERALIZATION: Accuracy: ' num2str(sum(zvalid)/numel(zvalid))])
%% 'Check OG data again'
[Xcheck, hcheck, ~, zcheck, ~, ~, cc_rate_fin, fa_rate_fin] = mklsrcUpdate(Hfull, Gfull, Bfull, etatest, trainClassSmall, classes, num_per_class);
display(['TRAINING: 2nd check'])
validacc2 = sum(zcheck)/numel(zcheck);
display(['TRAINING: Accuracy: ' num2str(sum(zcheck)/numel(zcheck))])
%% Make some figures
figure(919); clf; hold on
text_vec = 1:500:size(fa_rate_fin, 2);
for i=1:size(cc_rate_fin,1)
    color = [mod((3*i/15),1) mod((6*(i+1)/15),1) 1-(i/10)];
    bb(i) = plot(fa_rate_fin(i,:), cc_rate_fin(i,:),'--', 'Color', color);
%     text(results.fa_rate_nonan(i, text_vec), results.cc_rate_nonan(i, text_vec), num2str(i-1));
    c{i} = [num2str(i-1) ''];
end
% ylim([.75 1]);
grid on;
legend(bb,c)
ylabel('Probability of correct classification')
xlabel('Probability of false alarm')


fa_rate_fin_avg = nanmean(fa_rate_fin);
cc_rate_fin_avg = nanmean(cc_rate_fin);
fa_rate_ini_avg = nanmean(fa_rate_ini);
cc_rate_ini_avg = nanmean(cc_rate_ini);
figure(906); clf
plot(fa_rate_ini_avg,cc_rate_ini_avg,'LineWidth',2);
hold on;
plot(fa_rate_fin_avg,cc_rate_fin_avg,'r','LineWidth',2)
xlabel('P_{FA}')
ylabel('P_{CC}')
legend('After Baseline Training','After In-Situ Learning')
grid on

figure(48); clf; hold on
plot( mean(z,2), '--');
plot( mean(zvalid,2)); 
legend('Test','Generalization')
% xlim([1 length(results.AUC_gen)])
xlabel('In-Situ Learning Iteration'); ylabel('Correct Classification Rate')
%% Look at the recon errors
% FIXME - look at only the reconstruction from the class it was classed as
doanal = 0;
if doanal
    for i=1:length(testClassSmall)
        Xtemp = Xtest(:,i);
        Xtemp(dictClassSmall ~= htest(i)) = 0;
        Xtest_classonly(:,i) = Xtemp;
    end
    
    trainrecon_errors = trainSetSmall - Dict*X;
    trainerrornorms = vecnorm(trainrecon_errors);
    
    testrecons = Dict*Xtest_classonly;
    testrecon_errors = testSetSmall - testrecons;
    testerrornorms = vecnorm(testrecon_errors);
    
    for i=1:length(testClassSmall)
        immmsestest(i) = immse(testrecons(:,i), testSetSmall(:,i));
        ssimstest(i) = ssim(reshape(testrecons(:,i),28,28), reshape(testSetSmall(:,i),28,28));
    end
end