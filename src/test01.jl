ENV["DISPLAY"]="localhost:12.0"
using GLMakie
using Random
using Random:seed!
GLMakie.activate!()

function lines_in_3D()
    seed!(123)
    n = 10
    x, y, z = randn(n), randn(n), randn(n)
    fig = Figure(; resolution=(1200, 500))
    ax1 = Axis3(fig[1, 1]; aspect=(1, 1, 1), perspectiveness=0.5)
    ax2 = Axis3(fig[1, 2]; aspect=(1, 1, 1), perspectiveness=0.5)
    ax3 = Axis3(fig[1, 3]; aspect=:data, perspectiveness=0.5)
    lines!(ax1, x, y, z; color=1:n, linewidth=3)
    scatterlines!(ax2, x, y, z; markersize=15)
    hm = meshscatter!(ax3, x, y, z; markersize=0.2, color=1:n)
    lines!(ax3, x, y, z; color=1:n)
    Colorbar(fig[2, 1], hm; label="values", height=15, vertical=false,
        flipaxis=false, ticksize=15, tickalign=1, width=Relative(3.55 / 4))
    fig
end

lines_in_3D()