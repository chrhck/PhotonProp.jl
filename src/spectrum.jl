
module Spectral

export Spectrum, Monochromatic, CherenkovSpectralDist, CherenkovSpectrum
export frank_tamm, frank_tamm_norm, frank_tamm_inverted_cdf

using Distributions

using Interpolations
using PhysicalConstants.CODATA2018
using Unitful
using Random
using StaticArrays
using LinearAlgebra

using ..Medium
using ..Utils

abstract type Spectrum end

struct Monochromatic{T} <: Spectrum
    wavelength::T
end



"""
    frank_tamm(wavelength::Real, ref_index::T) where {T<:Real}

Evaluate Frank-Tamm formula
"""
function frank_tamm(wavelength::Real, ref_index::T) where {T<:Real}
    return T(2 * pi * FineStructureConstant / wavelength^2 * (1 - 1 / ref_index^2))
end

"""
    frank_tamm_norm(wl_range::Tuple{T, T}, ref_index_func::Function) where {T<:Real}

Calculate number of Cherenkov photons per length in interval `wl_range`.
Returned number is in units cm^-1.
"""
function frank_tamm_norm(wl_range::Tuple{T,T}, ref_index_func::Function) where {T<:Real}
    f(x) = frank_tamm(x, ref_index_func(x))
    integrate_gauss_quad(f, wl_range[1], wl_range[2]) * 1E7
end


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

    CherenkovSpectrum(::Tuple{T,T}, ::SVector{N,T}) where {T<:Real,N<:Integer} = error("default constructor disabled")

    function CherenkovSpectrum(wl_range::Tuple{T,T}, interp_steps::U, medium::V) where {T<:Real,U<:Integer,V<:MediumProperties}
        spec = CherenkovSpectralDist(wl_range, medium)
        eval_knots = range(T(0), T(1), interp_steps)
        knots = SVector{interp_steps,T}(spec.interpolation(eval_knots))
        return new{T,interp_steps}(wl_range, knots)
    end

end

end


