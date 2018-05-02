function K=computeMultiKernelMatrixFromPrecomputed(kernels,eta)
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
    if eta(i) ~= 0
        K = K + eta(i)*kernels{i};
    end
end

end