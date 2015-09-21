function plotOpinions(opinions)
% PLOT_OPINIONS Plot opinions graph (without equilibrium)
    t = size(opinions,2);
    figure;
    plot(1:t,opinions(:,1:t));
    ylabel('Opinions');
    xlabel('t');
    axis([1, t, max([0 min(opinions(:))-0.2]), min([1 max(opinions(:)) + 0.2])]);
    grid on;
end

