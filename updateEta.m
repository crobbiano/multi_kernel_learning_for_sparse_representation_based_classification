function [eta, err] = updateEta(eta, c, mu, z, zm)
%updateEta Updates the mixing weights of the kernels
%   Detailed explanation goes here
% [best_c_val,best_c_idx] = max(c .* double(eta==0)); % THIS WORKS
[best_c_val,best_c_idx] = max(c); % THIS WORKS
best_c_val_og = best_c_val;
update_c = 1;
if update_c
    for i=best_c_idx:-1:1
        if (c(i) ~=0) & (c(i) + mu > best_c_val_og)  % THIS WORKS
            best_c_idx = i;
            best_c_val = c(i);
%             display(['Changed best_c to higher index: ' num2str(i)])
        end
    end
end
new_kernel = best_c_idx;
new_kernel_weight = sum(bitand(zm(new_kernel,:), not(z)))/sum(bitor(not(z), not(zm(new_kernel,:))));
curr_kernel_weight = sum(bitand(z, not(zm(new_kernel,:))))/sum(bitor(not(z), not(zm(new_kernel,:))));
prev_eta = eta; % save for calcing error
% Do the actual mixing weight update here
for m=1:length(eta)
    if m==new_kernel
%         if eta(m) ~=0 % If the new kernel is already part of the curr kernel
%             eta(m)= new_kernel_weight + eta(m)*curr_kernel_weight;
%         else
            eta(m) = new_kernel_weight;
%         end
    else
        eta(m) = eta(m)*curr_kernel_weight;
    end
end

% Compute sums of all weights and normalize weights by sum
total_weights = sum(eta);
eta = eta/total_weights;
err = norm(prev_eta - eta,2);
end

