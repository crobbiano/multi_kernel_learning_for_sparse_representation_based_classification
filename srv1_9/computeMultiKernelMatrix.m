function K=computeMultiKernelMatrix(A,B,eta,kfncs)
% Compute the multi kernel matrix, K=eta*kernel(A,B)
% A: matrix, each column is a sample
% B: matrix, each column is a sample
% eta: weights for each kernel function
% kfnc: anonymous functions for kernels
% option: struct, include files:
% option.kernel: string, can be 'linear','polynomial','rbf','sigmoid','ds'
% option.param
% K: the kernel matrix
% Yifeng Li, September 03, 2011

K = 0;
for i=1:length(eta)
    option.kernel = 'cust'; option.kernelfnc=kfncs{i};
    if eta(i) ~= 0
        K = K + eta(i)*computeKernelMatrix(A, B, option);
    end
end

end