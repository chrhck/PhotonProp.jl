
module Spectral

export Spectrum, Monochromatic, CherenkovSpectralDist, CherenkovSpectrum
export frank_tamm, frank_tamm_norm, frank_tamm_inverted_cdf

using Distributions
using QuadGK
using Interpolations
using PhysicalConstants.CODATA2018
using Unitful
using Random
using StaticArrays

using ..Medium

abstract type Spectrum end

struct Monochromatic{T} <: Spectrum
    wavelength::T
end


"""
    frank_tamm(Length, RefIndex, [Permeability])

Evaluate Frank-Tamm formula
"""
function frank_tamm(wavelength::Unitful.Length{T}, ref_index::T) where {T<:Real}
    return T(2 * pi * FineStructureConstant / wavelength^2 * (1 - 1 / ref_index^2))
end

"""
    frank_tamm_norm(wl_range, ref_index_func)

Calculate number of Cherenkov photons in interval `wl_range`.
If `wl_range` is unitles, assumes it's given in nm
"""
function frank_tamm_norm(wl_range::Tuple{Unitful.Length{T},Unitful.Length{T}}, ref_index_func::Function) where {T<:Real}
    quadgk(x -> frank_tamm(x, ref_index_func(x)), wl_range[1], wl_range[2])[1] * 1u"cm" |> NoUnits
end

frank_tamm_norm(wl_range::Tuple{Real,Real}, ref_index_func::Function) = frank_tamm_norm(wl_range .* 1u"nm", ref_index_func)

"""
    frank_tamm_inverted_cdf(wl_range, step_size)

Return the inverted CDF of the Frank-Tamm Spectrum in range `wl_range` evaluated with
step size `step_size`.
"""
function frank_tamm_inverted_cdf(wl_range::Tuple{T,T}, medium::MediumProperties, step_size::T=T(1)) where {T}
    wl_steps = wl_range[1]:step_size:wl_range[2]

    norms = Vector{T}(undef, size(wl_steps, 1))
    norms[1] = 0

    full_norm = frank_tamm_norm(wl_range, wl -> get_refractive_index(wl, medium))

    for i in 2:size(wl_steps, 1)
        step = wl_steps[i]
        norms[i] = frank_tamm_norm((wl_range[1], step), wl -> get_refractive_index(wl, medium)) / full_norm
    end

    sorting = sortperm(norms)

    return norms[sorting], wl_steps[sorting]
end


struct CherenkovSpectralDist <: Sampleable{Univariate,Continuous}
    interpolation
    wl_range::Tuple{Float64,Float64}

    function CherenkovSpectralDist(wl_range::Tuple{T,T}, medium::MediumProperties) where {T<:Real}

        norms, wl_steps = frank_tamm_inverted_cdf(wl_range, medium)
        p = LinearInterpolation(norms, wl_steps)
        new(p, wl_range)
    end

end

Base.:rand(rng::AbstractRNG, s::CherenkovSpectralDist) = s.interpolation(rand(rng))



struct CherenkovSpectrum{T<:Real,N} <: Spectrum
    wl_range::Tuple{T,T}
    knots::SVector{N,T}

    Cherenkov(::Tuple{T,T}, ::SVector{N,T}) where {T<:Real,N<:Integer} = error("default constructor disabled")

    function Cherenkov(wl_range::Tuple{T,T}, interp_steps::U, medium::V) where {T<:Real,U<:Integer,V<:MediumProperties}
        spec = CherenkovSpectralDist(wl_range, medium)
        eval_knots = range(T(0), T(1), interp_steps)
        knots = SVector{interp_steps,T}(spec.interpolation(eval_knots))
        return new{T,interp_steps}(wl_range, knots)
    end

    function Cherenkov(
        wl_range::Tuple{Unitful.Length{T},Unitful.Length{T}},
        interp_steps::U,
        medium::V) where {T<:Real,U<:Integer,V<:MediumProperties}
        Cherenkov((ustrip(u"nm", wl_range[1]), ustrip(u"nm", wl_range[2])), interp_steps, medium)
    end

end

end


