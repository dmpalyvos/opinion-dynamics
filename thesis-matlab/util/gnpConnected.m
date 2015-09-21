function [A] = gnpConnected(n,p)
%GNP Create a g(n,p) random network
%   INPUT:
%       [n] =  Number of nodes
%       [p] =  Probability of each edge being created
%   OUTPUT:
%       [A] =  nxn adjacency matrix (random weights)

% Init
A = zeros(n);
% Ensure connectivity
A = double(A | randomSpanningTree(n));
for i = 1:n
    for j = 1:i
        r = rand();
        if (r < p)
            w = rand();
            A(i,j) = w;
            A(j,i) = w;
        end
    end
end

fprintf(1, 'Created Connected g(n,p) network | n = %d | p = %4.3f | Mean Degree = %3.2f\n', n, p, meanDegree(A));


end

