function A = randomSpanningTree(N)
% randomSpanningTree - Create a graph with N nodes 
% connected by a random spanning tree


A = zeros(N);
nodeList = randperm(N);

for k = 1:(N-1)
    i = nodeList(k);
    j = nodeList(k+1);
    A(i,j) = 1;
    A(j,i) = 1;
end

end

