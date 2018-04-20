function [x] = getCoeffs(prev_x, y, A, kappa, lambda, eta, num_iter, train_idx)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    
    % compute the current kernel based on weights
    curr_kernel = zeros(size(A,2));
    % Some tom-foolry with the partial kernel to make it work with under lying operations
    % stick the sample, y, in the first column of a matrix of size A and zero out everying
    % else.  then after the computation is done, extract the first column again
    partial_kernel = zeros(size(A,2));
%     partial_kernel = zeros(1, size(A,2));
    single_kernel = 0;
    
    y_mat = zeros(size(A));
    y_mat(:,1) = y;
    
    for i=1:length(kappa)
        option.kernel = 'cust'; option.kernelfnc=kappa{i};
        curr_kernel = curr_kernel + eta(i)*computeKernelMatrix(A,A,option);
        partial_kernel = partial_kernel + eta(i)*computeKernelMatrix(y_mat,A,option);
        single_kernel = single_kernel + eta(i)*computeKernelMatrix(y,y,option);
    end
    
    partial_kernel = partial_kernel(1,:);
    
    
    %     for m=1:length(kappa)
    %         K = zeros(size(A,2), size(A,2));
    %         for i=1:size(A,2)
    %             for j=1:size(A,2)
    %                 K(i,j) = eta(m)*kappa{m}(A(:,i), A(:,j));
    %             end
    %             partial_kernel(i) = partial_kernel(i) + eta(m)*kappa{m}(y, A(:,i));
    %         end
    %         curr_kernel = curr_kernel + K;
    %         single_kernel = single_kernel + eta(m)*kappa{m}(y, y);
    %     end
    
    if train_idx > 0
        partial_kernel(:, train_idx) = 0;
        curr_kernel(:, train_idx) = 0;
        curr_kernel(train_idx, :) = 0;
    end
    
    option.lambda = lambda;
    option.iter = num_iter;
%     option.SCMethod='l1qpAS';
    x = KSRSC(curr_kernel, partial_kernel', single_kernel, option);
end

