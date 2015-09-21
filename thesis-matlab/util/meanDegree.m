function d = meanDegree(A)
%MEANDEGREE Calculate the mean degree of a graph given an adjacency matrix A

A(A>0) = 1;
degrees = sum(A,2);
d = mean(degrees);

end

