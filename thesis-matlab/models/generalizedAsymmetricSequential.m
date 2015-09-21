function [At, opinions] = generalizedAsymmetricSequential(A, sB, s, tMax, c_eps, varargin)
% generalizedAsymmetricSequential  - Simulate the GA (or variable weights model)
%                                    SEQUENTIAL Updating
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
%       [At] = Not used
%       [opinions] = Nxt matrix of the opinions at every round

wantPlot = false;
testStability = false;
STABILITY_EPS = 1e-4;
NORM_TYPE = Inf;

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

N = size(A,1);
A = A - diag(diag(A)); % Remove stubborness from A


% Run the simulation
z = s;
zPrev = s;
opinions = zeros(N,tMax+1);
opinions(:,1) = z;
sB = diag(sB); % Stubborness
update_order = randperm(N);
moved = false;
At = [];
for t = 1:tMax
    i = update_order(mod(t,N)+1);
    Q = varWeight(A, s, z) + sB;
    [~, Q, B, ~] = preprocessGraph(Q, s);
    Q = Q - B;
    % Update node
    z(i) = B(i,i)*s(i) + Q(i,:)*zPrev;
    zPrev = z;
    opinions(:,t+1) = z;
    % Check if we have reached an equilibrium
    opinionDiff = diff(opinions(:,t:t+1),1,2);
    if (norm(opinionDiff, NORM_TYPE) < c_eps)
        disp(['[GA Sequential] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    else if (testStability && norm(opinionDiff, NORM_TYPE) < STABILITY_EPS) % Fluctuate opinions
            if (moved)
                continue;
            end
            moved = true;
            z = z + rand(N,1) - 0.5;
            z(z>1) = 1.0;
            z(z<0) = 0.0;
            opinions(:,end) = z;
        end
    end
end
opinions(:,t+2:end) = [];


% Output
if (wantPlot)
    plotOpinions(opinions);
end

end
