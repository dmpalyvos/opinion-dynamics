function [equilibrium, opinions] = randomParametric(A, c, s, tMax, c_eps, varargin)
% randomParametric - Simulate the 'Friend meetup' Model with the additional
% parameter c (á in the text)
%
%   INPUT:
%       [A] = NxN Adjacency Matrix of the Network
%       [c] = scalar, defines the behaviour of the model
%       [s] = Nx1 Vector of the Intrisic Beliefs
%       [tMax] = Maximum number of rounds
%       [c_eps] = Convergence limit
%       Other arguments:
%           'plot' = Plot Opinions vs Time
%
%   OUTPUT:
%       [equilibrium] = The expected equilibrium
%       [opinions] = Nxt matrix of the opinions at every round

% Init
[N, A, ~, equilibrium] = preprocessGraph(A, s);

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

% Run the simulation
x = s;
x_prev = x;
opinions = zeros(N,tMax+1);
opinions(:,1) = x;
for t = 1:tMax
    % Update average opinion for every node
    op = zeros(N,1);
    for i = 1:N
        Ai = rdecide(A(i,:)); % Choose a random node
        if (Ai == -1)
            continue;
        end
        if (Ai == i)
            op(i) = s(i);
        else
            op(i) = x(Ai);
        end
    end
    x = (1-c)*x_prev + c*op;
    x_prev = x;
    opinions(:,t+1) = x;
    % Check if we have reached the equilibrium
    if (max(abs(x-equilibrium)) < c_eps)
        disp(['[Random param] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end

% Output
if (wantPlot)
    plotOpinions2(opinions);
end

end
