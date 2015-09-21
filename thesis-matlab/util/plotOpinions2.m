function plotOpinions2(opinions)
% plotOpinions2 - Plot opinions graph (without equilibrium)
    N = size(opinions,1);
    t = size(opinions,2);
    figure;
    for i = 1:N
        color_line(1:t, opinions(i,1:t), opinions(i,1:t));
        hold on;
    end
    ylabel('Opinions');
    xlabel('t');
    axis([1, t, max([0 min(opinions(:))-0.2]), min([1 max(opinions(:)) + 0.2])]);
    grid on;
end

