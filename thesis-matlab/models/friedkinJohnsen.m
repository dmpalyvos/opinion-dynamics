function [equilibrium, opinions] = friedkinJohnsen(A, s, tMax, c_eps, varargin)
% friedkinJohnsen - Simulate the Friedkin-Johnsen Model
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
%       [equilibrium] = The expected equilibrium
%       [opinions] = Nxt matrix of the opinions at every round

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
[N, A, B, equilibrium] = preprocessGraph(A, s);

% Zero out the diagonal of A
A = A - B; 

% Run the simulation
opinions = zeros(N, tMax);
x = s;
x_prev = x;
opinions(:,1) = x;
for t = 1:tMax
    x = A*x_prev + B*s; % Update all nodes
    x_prev = x;
    opinions(:, t+1) = x;
    % Check if we have reached the equilibrium
    if (max(abs(x-equilibrium)) < c_eps)
        disp(['[Friedkin-Johnsen] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end

opinions(:,t+2:end) = [];

% Plot
if (wantPlot)
    plotOpinions2(opinions);
end

end
