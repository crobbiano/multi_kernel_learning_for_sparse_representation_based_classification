function [ alignment ] = kernelAlignment( K1, K2 )
    %kernelAlignment Calculates the kernel alignment between two kernel mats
    %   K1 and K2 must be the same size.  Calculates the Frobenious norm 
    %   between the two matrices, divided by the squareroot of the product
    %   of the Frobenious norm of each kernel mat with itself.
    
    alignment = trace(K1'*K2)/sqrt(trace(K1'*K1)*trace(K2'*K2));

end

