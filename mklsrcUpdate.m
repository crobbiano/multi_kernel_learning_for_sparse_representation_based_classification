function [X, h, g, z, zm, c, cc_rate, fa_rate] = mklsrcUpdate(Hfull, Gfull, Bfull, eta, trainClassSmall, classes, num_per_class)
%mklsrcUpdate Do an update for MKL-SRC
%   Detailed explanation goes here

optionKSRSC.lambda=0.1;
% optionKSRSC.SCMethod='l1qpAS'; % can be nnqpAS, l1qpAS, nnqpIP, l1qpIP, l1qpPX, nnqpSMO, l1qpSMO
optionKSRSC.iter=200;
optionKSRSC.dis=0;
optionKSRSC.residual=1e-4;
optionKSRSC.tof=1e-4;


H=computeMultiKernelMatrixFromPrecomputed(Hfull,eta);

textprogressbar('.....Coeff progress: ');
for idx = 1:length(trainClassSmall)
    % compute kernels
    G=computeMultiKernelMatrixFromPrecomputed(Gfull(:, idx),eta);
    B=computeMultiKernelMatrixFromPrecomputed(Bfull(:, idx),eta);
    
    % KSRSC sparse coding
        [X(:, idx), ~, ~] =KSRSC(H,G,diag(B),optionKSRSC);
%     X(:,idx) = OMP(H, G, 45, .0001);
    
    % Find class - calculate h (class) and z (correct class)
    classerr = zeros(1, length(classes));
    for class=1:length(classes)
        b_idx = sum(num_per_class(1:class-1)) + 1;
        e_idx = sum(num_per_class(1:class));
        x_c = X(b_idx:e_idx ,idx);
        kernel_c = H(b_idx:e_idx ,b_idx:e_idx);
        partial_c = G(b_idx:e_idx)';
        classerr(class) = B + x_c'*kernel_c*x_c - 2*partial_c*x_c;
    end
    retrieve_score(idx, :) = classerr;
    [~, h(idx)] = min(classerr);
    h(idx)=classes(h(idx));
    z(idx) = (h(idx) == trainClassSmall(idx));
    
    % Need to calculate the ability to classify for each individual kernel
    for kidx=1:length(eta)
        eta_temp = [];
        eta_temp(kidx) = 1; % place a 1 in the current kernel
        
        H_temp = computeMultiKernelMatrixFromPrecomputed(Hfull, eta_temp);
        G_temp = computeMultiKernelMatrixFromPrecomputed(Gfull(:, idx),eta_temp);
        B_temp = computeMultiKernelMatrixFromPrecomputed(Bfull(:, idx),eta_temp);
        
        err_temp = zeros(1, length(classes));
        for class=1:length(classes)
            b_idx = sum(num_per_class(1:class-1)) + 1;
            e_idx = sum(num_per_class(1:class));
            x_c = X(b_idx:e_idx ,idx);
            kernel_c = H_temp(b_idx:e_idx ,b_idx:e_idx);
            partial_c = G_temp(b_idx:e_idx)';
            err_temp(class) = B_temp + x_c'*kernel_c*x_c - 2*partial_c*x_c;
        end
        [~, h_temp] = min(err_temp);
        h_temp = classes(h_temp);
        g(kidx, idx) = h_temp;
        zm(kidx, idx) = g(kidx, idx) == trainClassSmall(idx);
        
        if sum(1-z)
            c(kidx,1) = sum(zm(kidx, find(z==0)))/sum(1-z);
        else
            c(kidx,1) = 0;
        end
        

    end
    
    textprogressbar(idx*100/length(trainClassSmall));
end

thresh = linspace(0,max(max(retrieve_score)),5000);
cc_rate = zeros(length(classes), length(thresh));
fa_rate = zeros(length(classes), length(thresh));
for label_idx=1:length(classes)
    for i = 1:length(thresh)
        cc_rate(label_idx, i) = length(find(retrieve_score(logical(trainClassSmall==classes(label_idx)),label_idx)<=thresh(i)))/...
            length(retrieve_score(logical(trainClassSmall==classes(label_idx))));
        fa_rate(label_idx, i) = length(find(retrieve_score(~logical(trainClassSmall==classes(label_idx)),label_idx)<=thresh(i)))/...
            length(retrieve_score(~logical(trainClassSmall==classes(label_idx))));
    end
end
textprogressbar(' ');
end

