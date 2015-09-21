function [opinions,Q] = hkGlobal(s, opEps, tMax, c_eps, varargin)
% hkGlobal  - Simulate the Hegselmann-Krause Model
%
%   INPUT:
%       [s] = Nx1 Vector of the Intrisic Beliefs
%       [opEps] = Scalar, å parameter of the model (confidence)
%       [tMax] = Maximum number of rounds
%       [c_eps] = Convergence limit
%       Other arguments:
%           'plot' = Plot Opinions vs Time
%
%   OUTPUT:
%       [opinions] = Nxt matrix of the opinions at every round
%       [Q] = NxN Adjacency Matrix at the end of the simulation

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

N = size(s,1);
% Run the simulation
z = s;
zPrev = z;
opinions = s;
for t = 1:tMax
    Q = zeros(N);
    for i = 1:N
        Ni = abs(zPrev - zPrev(i)) <= opEps; % Neighbors of i at step t
        Q(i,Ni) = 1;
        z(i) = mean(zPrev(Ni));
    end
    zPrev = z;
    opinions = [opinions z];
    % Check if we have reached an equilibrium
    opinionDiff = diff(opinions(:,end-1:end),1,2);
    if (norm(opinionDiff, NORM_TYPE) < c_eps)
        disp(['[HK Global] Reached equilibrium after ' num2str(t) ' rounds.']);
        break;
    end
end

% Output
if (wantPlot)
    plotOpinions2(opinions);
end

end
