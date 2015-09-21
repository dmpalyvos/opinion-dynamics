function [convTime, finalOpinions] = kNN2_NOMEM(A, s, K, tMax, c_eps)
% kNN2_NOMEM - Simulate the K-Nearest Neighbors Model (Dynamic)
%              Fast Simulation with minimal memory consumption 
%              No Graphs
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
    % Find all paths distance 2
    A2 = A^2 | A; % is | A needed?
    for i = 1:N
        neighbors2 = find(A2(i,:));
        neighborsDist = abs(zPrev(neighbors2) - zPrev(i));
        [~, sortedNeighbors] = sort(neighborsDist, 'ascend');
        % If the node has L < K connections, he will have L friends
        L = min(K,numel(neighbors2));
        friends = neighbors2(sortedNeighbors(1:L));
        Q(i,friends) = 1/L;
    end
    A = Q;
    z = A*zPrev; % z(t+1) = A(t)z(t)
    % Check if we have reached an equilibrium
    opinionDiff = z - zPrev;
    zPrev = z;
    if (norm(opinionDiff, Inf) < c_eps)
        %disp(['[HK Local K-NN2] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end

convTime = t;
finalOpinions = z;

end
