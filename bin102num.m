function [ num ] = bin102num( binnum )
%num2bin10 Converts 0-9 to 10 digit fake binary
% num array of numbers

num = zeros(1, size(binnum, 2));

for i = 1: size(binnum, 2)
    num(i) = find(binnum(:,i) == 1);
    num(i) = num(i);
end

end

