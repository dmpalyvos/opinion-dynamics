function E = rdecide(probability)
% rdecide - Choose an event with a specified probability
%
%   INPUT:
%       [probability] = 1xN vector with the probabilities [0,1] 
%       of each event
%
%   OUTPUT:
%       [D] = scalar representing the event chosen 
%       in the range 1:numel(probability)

if (numel(probability) < 1)
    error('rdecide: Empty event vector')
end

% Bound check
if (size(probability,1) > 1)
    error('rdecide: Wrong matrix dimensions');
end

% Init
positiveP = find(probability);
n = numel(positiveP);
s = 0;
r = rand();
E = -1;
% Choose event according to probability vectors
for i = positiveP
    s = s + probability(i);
    if (r <= s)
        E = i;
        return;
    end
end

if (E == -1) % Should happen only if we get a vector consisting of zeros
    warning('rdecide: Something went really wrong | returning -1');
end

end

