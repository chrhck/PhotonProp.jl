module LightYield

export LongitudinalParametersEMinus, LongitudinalParametersEPlus, LongitudinalParametersGamma
export MediumPropertiesWater, MediumPropertiesIce
export CherenkovTrackLengthParametersEMinus, CherenkovTrackLengthParametersEPlus, CherenkovTrackLengthParametersGamma
export longitudinal_profile, cherenkov_track_length, cherenkov_counts

using Parameters:@with_kw
using SpecialFunctions:gamma
using StaticArrays
using QuadGK
using Unitful
using PhysicalConstants.CODATA2018
using ..PhotonSource


@with_kw struct MediumProperties
    ref_ix::Float64
    density::Float64
    rad_len::Float64
end

MediumPropertiesWater = MediumProperties(ref_ix=1.333, density=1.0u"g/cm^3", rad_len=36.08u"g*cm^2")
MediumPropertiesIce = MediumProperties(ref_ix=1.309, density=0.9180"g/cm^3", rad_len=36.08u"g*cm^2")


@with_kw struct LongitudinalParameters
    alpha::Float64
    beta::Float64
    b::Float64
end

LongitudinalParametersEMinus = LongitudinalParameters(alpha=2.01849, beta=1.45469, b=0.63207)
LongitudinalParametersEPlus = LongitudinalParameters(alpha=2.00035, beta=1.45501, b=0.63008)
LongitudinalParametersGamma = LongitudinalParameters(alpha=2.83923, beta=1.34031, b=0.64526)


@with_kw struct CherenkovTrackLengthParameters
    alpha::Float64
    beta::Float64
    alpha_dev::Float64
    beta_dev::Float64
end

CherenkovTrackLengthParametersEMinus = CherenkovTrackLengthParameters(
    alpha=532.07078881,
    beta=1.00000211,
    alpha_dev=5.78170887,
    beta_dev=0.5
)

CherenkovTrackLengthParametersEPlus = CherenkovTrackLengthParameters(
    alpha=532.11320598,
    beta=0.99999254,
    alpha_dev=5.73419669,
    beta_dev=0.5
)

CherenkovTrackLengthParametersGamma = CherenkovTrackLengthParameters(
    alpha=532.08540905,
    beta=0.99999877,
    alpha_dev=5.78170887,
    beta_dev=5.66586567
)


parameter_a(energy::Real, alpha::Real, beta::Real) = alpha + beta * log10(energy)


parameter_a(
    energy::Unitful.Energy,
    long_pars::LongitudinalParameters
) = parameter_a(ustrip(u"GeV", energy), long_pars.alpha, long_pars.beta)


parameter_b(energy::Real, b::Real) = b

parameter_b(
    energy::Unitful.Energy,
    long_pars::LongitudinalParameters
    ) = parameter_b(ustrip(u"GeV", energy), long_pars.b)
    


function longitudinal_profile(energy::Real, z::Real, rad_len::Real, density::Real)

function longitudinal_profile(energy::Real, z::Real, medium::MediumProperties, long_pars::LongitudinalParameters) 
    lrad = medium.rad_len / medium.density 
    t = z / lrad
    b = parameter_b(energy, long_pars)
    a = parameter_a(energy, long_pars)
    
    b * ((t * b)^(a - 1.) * exp(-(t*b)) / gamma(a))
end

function fractional_contrib_long(
    energy::Real,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    long_pars::LongitudinalParameters,
    ) where {T <: Real}

    int_range = extrema(z_grid)
    norm = quadgk(
        z-> longitudinal_profile(energy, z, medium, long_pars),
        int_range[1], int_range[2])[1]

    part_contribs = Vector{Float64}(undef, size(z_grid, 1))
    part_contribs[1] = 0
    @inbounds for i in 1:size(int_grid, 1)-1
        lower = int_grid[i]
        upper = int_grid[i+1]
        part_contribs[i+1] = (
            1/norm * 
            quadgk(
                z-> longitudinal_profile(energy, z, medium, long_pars),
                lower, upper)[1]
        )
    end
    part_contribs
end


function cherenkov_track_length_dev(energy, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha_dev * energy^track_len_params.beta
end

function cherenkov_track_length(energy, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha * energy^track_len_params.beta 
end

function cherenkov_counts(wavelength::Real, track_length::Real, medium::MediumProperties)

    alpha_em = 0.0072973525693
    charge = 1.

    prefac = 2 * pi * alpha_em * charge^2 / (1 - 1 / medium.ref_ix^2)

    # 1e-7 due to the conversion from nm to cm
    diff_counts = prefac / (wavelength * 1e-9)^2. * track_length * 1e-2
    diff_counts * 1e-9 / pi
end


@enum ParticleType begin
    EMinus = 11
    EPlus = -11
    Gamma = 22
end

mutable struct Particle{T}
    position::SVector{3, T}
    time::T
    energy::T
    type::ParticleType
end

function make_elongated_cascade_sources(
    particle::Particle{T}
    range::Tuple{T, T},
    precision::T,
    medium::MediumProperties,
    long_params::LongitudinalParameters
    ) where {T <: Real}

    s = SobolSeq([range[1]], [range[2]])
    n_steps = Int64(ceil(range[2] - range[1]) / precision)
    int_grid = sort!(vec(reduce(hcat, next!(s) for i in 1:n_steps)))

    fractional_contrib = fractional_contrib_long(particle.energy, int_grid, medium, long_params)

    sources = Vector{PhotonSource}(undef, n_steps-1)

    pos_along = 0.5 .* (int_grid[1:end-1] .+ int_grid[2:end])

    for i in range 2:n_steps
        this_pos = particle.pos .+ pos_along .* particle.direction
        this_time
        
    
    end
end






end