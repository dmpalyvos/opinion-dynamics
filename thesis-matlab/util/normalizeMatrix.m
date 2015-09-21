function A = normalizeMatrix(A)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

W = sum(A,2); % Normalization Factor of each row

if (W > 0)
    for i = 1:size(A,1)
        A(i,:) = A(i,:) ./ W(i); % Normalize
    end
else 
    warning('Zero row in matrix A');
end
end

