function [ z ] = calcZis(x, y, A, kernels, kappa, eta, ls, train_idx, ker_num)
    %calcZis Calculates the zi's
    %   Find the current kernel which is a mixture of all ranked kernels
    %   then classifies all y using the current kernel and
    %   zi == 1 if classification is correct
    
    % compute the current kernel based on weights
    curr_kernel = zeros(size(A,2));
    % Some tom-foolry with the partial kernel to make it work with under lying operations
    % stick the sample, y, in the first column of a matrix of size A and zero out everying
    % else.  then after the computation is done, extract the first column again
    partial_kernel = zeros(size(A,2));
    %         partial_kernel = zeros(1, size(A,2));
    
    single_kernel = 0;
    y_mat = zeros(size(A));
    y_mat(:,1) = y;
    
    for i=1:length(kappa)
        if iscell(kappa)
            option.kernel = 'cust'; option.kernelfnc=kappa{i};
            if train_idx == 0
                curr_kernel = curr_kernel + eta(i)*computeKernelMatrix(A,A,option);
                partial_kernel = partial_kernel + eta(i)*computeKernelMatrix(y_mat,A,option);
                single_kernel = single_kernel + eta(i)*computeKernelMatrix(y,y,option);
            else
                curr_kernel = curr_kernel + eta(i)*kernels{i};
                partial_kernel = partial_kernel + eta(i)*kernels{i}(train_idx, :);
                single_kernel = single_kernel + eta(i)*kernels{i}(train_idx,train_idx);
            end
        else
            option.kernel = 'cust'; option.kernelfnc=kappa;
            if train_idx == 0
                curr_kernel = curr_kernel + eta(i)*computeKernelMatrix(A,A,option);
                partial_kernel = partial_kernel + eta(i)*computeKernelMatrix(y_mat,A,option);
                single_kernel = single_kernel + eta(i)*computeKernelMatrix(y,y,option);
            else
                curr_kernel = curr_kernel + eta(i)*kernels{ker_num};
                partial_kernel = partial_kernel + eta(i)*kernels{ker_num}(train_idx, :);
                single_kernel = single_kernel + eta(i)*kernels{ker_num}(train_idx,train_idx);
            end
        end
    end
    
    partial_kernel = partial_kernel(1,:);
    
    %     for m=1:length(kappa)
    %         K = zeros(size(A,2), size(A,2));
    %         for i=1:size(A,2)
    %             for j=1:size(A,2)
    %                 if iscell(kappa)
    %                     K(i,j) = eta(m)*kappa{m}(A(:,i), A(:,j));
    %                 else
    %                     K(i,j) = eta(m)*kappa(A(:,i), A(:,j));
    %                 end
    %             end
    %             if iscell(kappa)
    %                 partial_kernel(i) = partial_kernel(i) + eta(m)*kappa{m}(y, A(:,i));
    %             else
    %                 partial_kernel(i) = partial_kernel(i) + eta(m)*kappa(y, A(:,i));
    %             end
    %         end
    %         curr_kernel = curr_kernel + K;
    %         if iscell(kappa)
    %             single_kernel = single_kernel + eta(m)*kappa{m}(y, y);
    %         else
    %             single_kernel = single_kernel + eta(m)*kappa(y, y);
    %         end
    %     end
    
    % Find the number of y in each class
    classes = unique(ls);
    num_classes = numel(classes);
    for i=1:num_classes
        num_y_per_class(i) = sum(ls == classes(i));
    end
    
    % Find class
    z=zeros(1, size(y,2));
    for i=1:size(y,2)
        err = zeros(1, num_classes);
        for class=1:num_classes
            b_idx = sum(num_y_per_class(1:class-1)) + 1;
            e_idx = sum(num_y_per_class(1:class));
            x_c = x(b_idx:e_idx ,i);
            kernel_c = curr_kernel(b_idx:e_idx ,b_idx:e_idx);
            partial_c = partial_kernel(b_idx:e_idx);
            err(class) = single_kernel + x_c'*kernel_c*x_c - 2*partial_c*x_c;
        end
        [~, z(i)] = min(err);
    end
end

