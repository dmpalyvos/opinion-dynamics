function [maxNeighborDist, opinions, At, degrees] = kNN(A, s, K, tMax, c_eps, varargin)
% kNN - Simulate the K-Nearest Neighbors Model
%
%   INPUT:
%       [A] = NxN Adjacency Matrix of the Network
%       [s] = Nx1 Vector of the Intrisic Beliefs
%       [K] = Scalar, K parameter of the model
%       [tMax] = Maximum number of rounds
%       [c_eps] = Convergence limit
%       Other arguments:
%           'plot' = Plot Opinions vs Time
%           'plotDist' = Plot distance of the furthest neighbor of each
%           node over time
%           'plotDegrees' = Plot mean degree of the network over time
%
%   OUTPUT:
%       [maxNeighborDist] =  Nxt Distance of the furthest neighbor of each
%           node over time
%       [opinions] = Nxt matrix of the opinions at every round
%       [At] = NxN Adjacency Matrix at the end of the simulation
%       [degrees] = 1xt vector of the mean degree of the network over time


wantPlot = false;
plotDist = false;
plotDeg = false;


% Parse input
if (~isempty(varargin))
    for c=1:length(varargin)
        switch varargin{c}
            case {'plot'}
                wantPlot = true;
            case {'plotDist'}
                plotDist = true;
            case {'plotDegrees'}
                plotDeg = true;
            otherwise
                error(['Invalid optional argument, ', varargin{c}]);
        end % switch
    end % for
end % if

N = size(A,1);
% All nodes should be stubborn for the averaging to work
A = A - diag(diag(A));
A = A + eye(N);

% Run the simulation
z = s;
zPrev = z;
opinions = zeros(N,tMax+1);
opinions(:,1) = z;
maxNeighborDist = zeros(N,tMax);
degrees = zeros(1,tMax+1);
degrees(1) = meanDegree(A);
for t = 1:tMax
    Q = zeros(N);
    for i = 1:N
        % Distance from i: 1 Edge
        neighbors = find(A(i,:));
        neighborsDist = abs(zPrev(neighbors) - zPrev(i));
        [sortedDist, sortedNeighbors] = sort(neighborsDist, 'ascend');
        % If the node has L < K connections, he will have L friends
        L = min(K,numel(neighbors));
        friends = neighbors(sortedNeighbors(1:L));
        z(i) = mean(zPrev(friends));
        maxNeighborDist(i,t) = sortedDist(L);
        Q(i,friends) = 1;
    end
    degrees(t+1) = meanDegree(Q);
    zPrev = z;
    opinions(:,t+1) = z;
    % Check if we have reached an equilibrium
    opinionDiff = diff(opinions(:,t:t+1),1,2);
    if (norm(opinionDiff, Inf) < c_eps)
        disp(['[HK Local K-NN] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end
opinions(:,t+2:end) = [];
maxNeighborDist(:, t+1:end) = [];
At = Q;
degrees(t+2:end) = [];

if (wantPlot)
    plotOpinions2(opinions);
end

if (plotDist)
    figure;
    for i = 1:N
        color_line(1:t, maxNeighborDist(i,1:t), opinions(i,1:t));
        hold on;
    end
    ylabel('Max Distance');
    xlabel('t');
    axis([1, t, -inf, inf]);
    colorbar;
    hold off;
end

if (plotDeg)
    figure;
    plot(degrees);
    grid on;
    xlabel('t');
    ylabel('Degree');
    title(['K = ' num2str(K)]);
    axis([-inf,inf,0,max(degrees)+10]);
end

end
