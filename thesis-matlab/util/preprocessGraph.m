function [N, A, B, equilibrium] = preprocessGraph(A, s)
%PREPROCESSGRAPH Run checks and get basic values from social graph

N = size(A,1);
if (size(s,1) ~= N)
    error('Wrong opinion vector size');
end
A = normalizeMatrix(A);
B = diag(diag(A)); % Stubborness matrix

% Expected equilibrium from Kleinberg Model
equilibrium = inv(eye(N)-(A-B))*B*s;

end

