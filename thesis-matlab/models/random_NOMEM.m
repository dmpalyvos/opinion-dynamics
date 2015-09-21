function [tM] = random_NOMEM(A, s, tMax, c_eps)
% random - Simulate the 'Friend meetup' Model
%          Fast Simulation with minimal memory consumption 
%          No Graphs
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
%       [tM] = Scalar, Convergence Time


% WARNING: Matrix A should not have a row consisting of zeros
[N, A, ~, equilibrium] = preprocessGraph(A, s);

% Run the simulation
x = s;
x_prev = x;
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
    % Check if we have reached the equilibrium
    if (norm(x-equilibrium,Inf) < c_eps)
        disp(['[Random] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end
tM = t;

end
