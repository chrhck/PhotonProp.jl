module Utils
using FastGaussQuadrature
using LinearAlgebra

export fast_linear_interp, transform_integral_range
export integrate_gauss_quad

GL50 = gausslegendre(50)

function fast_linear_interp(x_eval::T, xs::AbstractVector{T}, ys::AbstractVector{T}) where {T}

    lower = first(xs)
    upper = last(xs)
    x_eval = clamp(x_eval, lower, upper)


    ix_upper = searchsortedfirst(xs, x_eval)
    ix_lower = ix_upper - 1

    @inbounds edge_l = xs[ix_lower]
    @inbounds edge_h = xs[ix_upper]

    step = edge_h - edge_l

    along_step = (x_eval - edge_l) / step

    @inbounds y_low = ys[ix_lower]
    @inbounds slope = (ys[ix_upper] - y_low)

    interpolated = y_low + slope * along_step

    return interpolated

end


function fast_linear_interp(x::T, knots::AbstractVector{T}, lower::T, upper::T) where {T}
    # assume equidistant

    x = clamp(x, lower, upper)
    range = upper - lower
    n_knots = size(knots, 1)
    step_size = range / (n_knots - 1)

    along_range = (x - lower) / step_size
    along_range_floor = floor(along_range)
    lower_knot = Int64(along_range_floor) + 1

    if lower_knot == n_knots
        return @inbounds knots[end]
    end

    along_step = along_range - along_range_floor
    @inbounds y_low = knots[lower_knot]
    @inbounds slope = (knots[lower_knot+1] - y_low)

    interpolated = y_low + slope * along_step

    return interpolated
end


function transform_integral_range(x, f, xrange)
    ba_half = (xrange[2] - xrange[1]) / 2

    u_traf = ba_half * x + (xrange[1] + xrange[2]) / 2
    f(u_traf) * ba_half

end

function integrate_gauss_quad(f, a, b, order::Union{Nothing,Integer}=nothing)
    if isnothing(order)
        nodes, weights = GL50
    else
        nodes, weights = gausslegendre(order)
    end
    dot(weights, map(x -> transform_integral_range(x, f, (a, b)), nodes))
end

end