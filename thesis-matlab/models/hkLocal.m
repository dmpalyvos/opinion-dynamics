function [At, opinions] = hkLocal(A, s, opEps, tMax, c_eps, varargin)
% hkLocal - Simulate the HK Model with limited knowledge
%
%   INPUT:
%       [A] = NxN Adjacency Matrix of the Network
%       [s] = Nx1 Vector of the Intrisic Beliefs
%       [opEps] = Scalar, å parameter of the model (confidence)
%       [tMax] = Maximum number of rounds
%       [c_eps] = Convergence limit
%       Other arguments:
%           'plot' = Plot Opinions vs Time
%
%   OUTPUT:
%       [At] = Not used
%       [opinions] = Nxt matrix of the opinions at every round

NORM_TYPE = Inf;

wantPlot = false;

% Parse input
if (~isempty(varargin))
    for c=1:length(varargin)
        switch varargin{c}
            case {'plot'}
                wantPlot = true;
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
At = [];
for t = 1:tMax
    for i = 1:N
        % Distance from i: 1 Edge
        neighbors = find(A(i,:));
        % Distance from i: 2 Edges
        neighbors2 = neighbors;
        neighbors2 = unique(neighbors2);
        neighborsDist = abs(zPrev(neighbors2) - zPrev(i));
        friends = neighbors2(neighborsDist <= opEps);
        z(i) = mean(zPrev(friends));
    end
    zPrev = z;
    opinions(:,t+1) = z;
    % Check if we have reached an equilibrium
    opinionDiff = diff(opinions(:,t:t+1),1,2);
    if (norm(opinionDiff, NORM_TYPE) < c_eps)
        disp(['[HK Local param å] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end
opinions(:,t+2:end) = [];

% Output
if (wantPlot)
    plotOpinions2(opinions);
end

end
