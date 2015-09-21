function [convTime, finalOpinions] = kNN_NOMEM(A, s, K, tMax, c_eps)
% kNN_NOMEM - Simulate the K-Nearest Neighbors Model 
%             Fast Simulation with minimal memory consumption 
%             No Graphs
%
%   INPUT:
%       [A] = NxN Adjacency Matrix of the Network
%       [s] = Nx1 Vector of the Intrisic Beliefs
%       [K] = Scalar, K parameter of the model
%       [tMax] = Maximum number of rounds
%       [c_eps] = Convergence limit
%
%   OUTPUT:
%       [convTime] = Scalar, Convergence Time
%       [finalOpinions] = Nx1 Final Opinions Vector


N = size(A,1);
% All nodes should be stubborn for the averaging to work
A = A - diag(diag(A));
A = A + eye(N);

% Run the simulation
z = s;
zPrev = z;
for t = 1:tMax
    Q = zeros(N);
    for i = 1:N
        % Distance from i: 1 Edge
        neighbors = find(A(i,:));
        neighborsDist = abs(zPrev(neighbors) - zPrev(i));
        [~, sortedNeighbors] = sort(neighborsDist, 'ascend');
        % If the node has L < K connections, he will have L friends
        L = min(K,numel(neighbors));
        friends = neighbors(sortedNeighbors(1:L));
        z(i) = mean(zPrev(friends));
        Q(i,friends) = 1;
    end
    % Check if we have reached an equilibrium
    opinionDiff = z - zPrev;
    zPrev = z;
    if (norm(opinionDiff, Inf) < c_eps)
        disp(['[K-NN] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end

convTime = t;
finalOpinions = z;

end
