function Q = varWeight(A, s, z)
%VARWEIGHT Get edge weights from a node to all its neighbors based on
%           the distance of their opinions
% INPUT:
%   i: 1x1 scalar Node id
%   A: NxN Adjacency matrix
%   s: Nx1 Intrisic beliefs
%   z: Nx1 Public opinions

N = size(A,1);
Q = zeros(N);
eps = 0.1;
p = 2;
e = exp(1);
for i = 1:N
    dist = abs(z-s(i));
    
    %c = 1 - dist;
    %c = 1 ./ log(dist + e);
    c = 1 ./ ((dist+eps).^p)
    
    q = zeros(1,N);
    neighbors = A(i,:) > 0;
    
    for j = find(neighbors)
        q(j) = c(j)/sum(c(neighbors));
    end
    Q(i,:) = q;
end

end

