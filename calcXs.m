function [ x_new ] = calcXs( etas, x, kernel_mats, Y, samples, ls )
    %calcZis Calculates the zi's
    %   Find the current kernel which is a mixture of all ranked kernels
    %   then classifies all samples using the current kernel and
    %   zi == 1 if classification is correct
      
    % get current kernel function
    curr_kernel = zeros(size(kernel_mats{1},1));
    for i=1:length(etas)
        curr_kernel = curr_kernel + etas(i)*kernel_mats{i};
    end
       
    % Find the number of samples in each class
    classes = unique(ls);
    num_classes = numel(classes);
    for i=1:num_classes
        num_samples_per_class(i) = sum(ls == classes(i));
    end
    
    % Find class
    x_new=zeros(size(x,2));
    for i=1:size(samples,2)
        kernel_copy = curr_kernel;
        kernel_copy(:,i) = 0; kernel_copy(i,:) = 0;
 
        err = [];
        for j=1:size(x, 2)
            err(j) = curr_kernel(i,i) + x(:,j)'*kernel_copy*x(:,j) - 2*kernel_copy(i,:)*x(:,j)
        end
        
        [~, idx] = min(err);
        x_new(:,i) = x(:,idx);
    end
    x_new;
end

