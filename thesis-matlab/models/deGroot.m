function [At, opinions] = deGroot(A, s, tMax, c_eps, varargin)
% deGroot - Simulate the DeGroot Model
%
%   INPUT:
%       [A] = NxN Adjacency Matrix of the Network
%       [s] = Nx1 Vector of the Intrisic Beliefs
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

% Init
[N, A, ~, ~] = preprocessGraph(A, s);

% Run the simulation
At = [];
z = s;
zPrev = z;
opinions = zeros(N,tMax+1);
opinions(:,1) = z;
for t = 1:tMax
    z = A * zPrev;
    zPrev = z;
    opinions(:,t+1) = z;
    % Check if we have reached an equilibrium
    opinionDiff = diff(opinions(:,t:t+1),1,2);
    if (norm(opinionDiff, NORM_TYPE) < c_eps)
        disp(['[DeGroot] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end

% Remove empty cells
opinions(:,t+2:end) = [];
% Output
if (wantPlot)
    plotOpinions(opinions);
end

end
