function [eigQ, opinions] = generalizedAsymmetric(A, sB, s, tMax, c_eps, varargin)
% generalizedAsymmetric - Simulate the GA (or variable weights model)
%
%   INPUT:
%       [A] = NxN Adjacency Matrix of the Network
%       [sB] = Nx1 The diagonal of the stubborness matrix B
%       [s] = Nx1 Vector of the Intrisic Beliefs
%       [tMax] = Maximum number of rounds
%       [c_eps] = Convergence limit
%       Other arguments:
%           'plot' = Plot Opinions vs Time
%           'testStability' = Shift the opinion vector at some point to
%           verify that the equilibrium is unique
%
%   OUTPUT:
%       [eigQ] = Average max eigenvalue of Q
%       [opinions] = Nxt matrix of the opinions at every round

NORM_TYPE = Inf;

wantPlot = false;
testStability = false;

% Parse input
if (~isempty(varargin))
    for c=1:length(varargin)
        switch varargin{c}
            case {'plot'}
                wantPlot = true;
                
            case {'testStability'}
                testStability = true;
                
            otherwise
                error(['Invalid optional argument, ', varargin{c}]);
        end % switch
    end % for
end % if

% If testStability is true then we will add some random fluctuation
% to z(t) when the difference between two consecutive rounds reaches
% the following value
STABILITY_EPS = 1e-2;


N = size(A,1);
A = A - diag(diag(A)); % Remove stubborness from A

% Run the simulation
z = s;
zPrev = z;
opinions = zeros(N,tMax+1);
opinions(:,1) = z;
sB = diag(sB); % Stubborness
moved = false; % For stability test
eigQ = 0;
%eT = [];
for t = 1:tMax
    Q = varWeight(A, s, z) + sB;
    [~, Q, B, ~] = preprocessGraph(Q, s);
    Q = Q - B;
    z = Q*zPrev + B*s; % Update all nodes
    zPrev = z;
    opinions(:,t+1) = z;
    eigQ = eigQ + max(eigs(Q));
    % Check if we have reached an equilibrium
    opinionDiff = diff(opinions(:,t:t+1),1,2);
    if (norm(opinionDiff, Inf) < c_eps)
        disp(['[Generalized Asymmetric] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    else if (testStability && norm(opinionDiff, NORM_TYPE) < STABILITY_EPS) % Fluctuate opinions
            if (moved)
                continue;
            end
            moved = true;
            z = z + rand(N,1) - 0.5;
            z(z>0.95) = 0.95;
            z(z<0.05) = 0.05;
            opinions(:,t+1) = z;
            zPrev = z;
        end
    end
end

% Remove empty cells
opinions(:,t+2:end) = [];

% Average Eigenvalue of Q
eigQ = eigQ / t;

% Plot
if (wantPlot)
    plotOpinions(opinions);
end

end
