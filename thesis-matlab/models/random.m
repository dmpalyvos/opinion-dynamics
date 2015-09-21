function [equilibrium, opinions] = random(A, s, tMax, c_eps, varargin)
% random - Simulate the 'Friend meetup' Model
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

% WARNING: Matrix A must not have a row consisting of zeros
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

[N, A, ~, equilibrium] = preprocessGraph(A, s);

% Run the simulation
x = s;
x_prev = x;
opinions = zeros(N,tMax+1);
opinions(:,1) = x;
for t = 1:tMax
    % Update average opinion for every node
    for i = 1:N
        if (all(A(i,:) == 0)) % rdecide will fail if all we have is 0s
            warning('[random] Row consists of zeros');
            continue;
        end
        Ai = rdecide(A(i,:)); % Choose a random node
        if (Ai == -1)
            error('[random] rdecide() returned -1, quiting...');
        end
        if (Ai == i)
            op = s(i);
        else
            op = x_prev(Ai);
        end
        x(i) = (op + t*x_prev(i)) / (t+1); % Calculate new opinion for node i
    end
    x_prev = x;
    opinions(:,t+1) = x;
    % Check if we have reached the equilibrium
    if (norm(x-equilibrium,Inf) < c_eps)
        disp(['[Random] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end
opinions(:,t+1:end) = [];

% Output
if (wantPlot)
    plotOpinions2(opinions);
end

end
