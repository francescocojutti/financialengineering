function plot_ellipse(mu, semi_axes_lengths, major_axis_direction)
    % Get semi-axes lengths
    a = semi_axes_lengths(1);
    b = semi_axes_lengths(2);

    % Get major axis direction
    theta = major_axis_direction;

    % Generate angles for plotting ellipse
    t = linspace(0, 2*pi, 100);

    % Calculate Cartesian coordinates of the ellipse points
    x = a * cos(t) * cos(theta) - b * sin(t) * sin(theta);
    y = a * cos(t) * sin(theta) + b * sin(t) * cos(theta);

    % Plot the ellipse
    plot(mu(1)+x, mu(2)+y, "Color","blue");
    xlabel('alpha');
    ylabel('tau');
    title('Plot of Ellipse');
end