module LightYield

export LongitudinalParameters, LongitudinalParametersEMinus, LongitudinalParametersEPlus, LongitudinalParametersGamma
export MediumPropertiesWater, MediumPropertiesIce
export CherenkovTrackLengthParameters, CherenkovTrackLengthParametersEMinus, CherenkovTrackLengthParametersEPlus, CherenkovTrackLengthParametersGamma
export longitudinal_profile, cherenkov_track_length, cherenkov_counts, fractional_contrib_long
export Particle, ParticleType, particle_to_lightsource, particle_to_elongated_lightsource

using Parameters: @with_kw
using SpecialFunctions: gamma
using StaticArrays
using QuadGK
using Sobol
using Zygote
using PhysicalConstants.CODATA2018
using Unitful
using ..Emission
using ..Spectral
using ..Medium
using ..Utils

c_vac_m_p_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)


@enum ParticleType begin
    EMinus = 11
    EPlus = -11
    Gamma = 22
end

@with_kw struct LongitudinalParameters
    alpha::Float64
    beta::Float64
    b::Float64
end

LongitudinalParametersEMinus = LongitudinalParameters(alpha=2.01849, beta=1.45469, b=0.63207)
LongitudinalParametersEPlus = LongitudinalParameters(alpha=2.00035, beta=1.45501, b=0.63008)
LongitudinalParametersGamma = LongitudinalParameters(alpha=2.83923, beta=1.34031, b=0.64526)

@with_kw struct CherenkovTrackLengthParameters
    alpha::Float64 # cm
    beta::Float64
    alpha_dev::Float64 # cm
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

@with_kw struct LightyieldParametrisation
    longitudinal::LongitudinalParameters
    track_length::CherenkovTrackLengthParameters
end

function get_params(ptype::ParticleType)
    if ptype == EPlus
        return LongitudinalParametersEPlus, CherenkovTrackLengthParametersEPlus
    elseif ptype == EMinus
        return LongitudinalParametersEMinus, CherenkovTrackLengthParametersEMinus
    elseif ptype == Gamma
        return LongitudinalParametersGamma, CherenkovTrackLengthParametersGamma
    end
end

get_longitudinal_params(ptype::ParticleType) = get_params(ptype)[1]
get_track_length_params(ptype::ParticleType) = get_params(ptype)[2]


function long_parameter_a_edep(
    energy::Real,
    long_pars::LongitudinalParameters
)
    long_pars.alpha + long_pars.beta * log10(energy)
end
long_parameter_a_edep(energy::Real, ptype::ParticleType) = long_parameter_a_edep(energy, get_longitudinal_params(ptype))

long_parameter_b_edep(::Real, long_pars::LongitudinalParameters) = long_pars.b
long_parameter_b_edep(energy::Real, ptype::ParticleType) = long_parameter_b_edep(energy, get_longitudinal_params(ptype))

"""
    longitudinal_profile(energy::Real, z::Real, medium::MediumProperties, long_pars::LongitudinalParameters)
    
energy in GeV, z in cm,
"""
function longitudinal_profile(
    energy::Real, z::Real, medium::MediumProperties, long_pars::LongitudinalParameters)

    unit_conv = 1000 # g/cm^2 / "kg/m^3" in cm    
    lrad = radiation_length(medium) / density(medium) * unit_conv # cm

    t = z / lrad
    b = long_parameter_b_edep(energy, long_pars)
    a = long_parameter_a_edep(energy, long_pars)

    b * ((t * b)^(a - 1.0) * exp(-(t * b)) / gamma(a))
end

function longitudinal_profile(
    energy, z, medium, ptype::ParticleType)
    longitudinal_profile(energy, z, medium, get_longitudinal_params(ptype))
end


function fractional_contrib_long!(
    energy::Real,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    long_pars::LongitudinalParameters,
    output::Union{Zygote.Buffer,AbstractVector{T}}
) where {T<:Real}

    if length(z_grid) != length(output)
        error("Grid and output are not of the same length")
    end

    int_range = extrema(z_grid)
    norm = integrate_gauss_quad(
        z -> longitudinal_profile(energy, z, medium, long_pars),
        int_range[1], int_range[2])

    output[1] = 0
    @inbounds for i in 1:size(z_grid, 1)-1
        output[i+1] = (
            1 / norm *
            integrate_gauss_quad(
                z -> longitudinal_profile(energy, z, medium, long_pars),
                z_grid[i], z_grid[i+1])[1]
        )
    end
    output
end

function fractional_contrib_long!(
    energy,
    z_grid,
    medium,
    ptype::ParticleType,
    output)
    fractional_contrib_long!(energy, z_grid, medium, get_longitudinal_params(ptype), output)
end

function fractional_contrib_long(
    energy::Real,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    pars_or_ptype::Union{LongitudinalParameters,ParticleType}
) where {T<:Real}
    output = similar(z_grid)
    fractional_contrib_long!(energy, z_grid, medium, pars_or_ptype, output)
end





function cherenkov_track_length_dev(energy::Real, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha_dev * energy^track_len_params.beta_dev
end
cherenkov_track_length_dev(energy::Real, ptype::ParticleType) = cherenkov_track_length_dev(energy, get_track_length_params(ptype))

function cherenkov_track_length(energy::Real, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha * energy^track_len_params.beta
end
cherenkov_track_length(energy::Real, ptype::ParticleType) = cherenkov_track_length(energy, get_track_length_params(ptype))

mutable struct Particle{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    energy::T
    type::ParticleType
end

function particle_to_lightsource(
    particle::Particle{T},
    medium::MediumProperties,
    wl_range::Tuple{T,T}
) where {T<:Real}

    total_contrib = (
        frank_tamm_norm(wl_range, wl -> medium.ref_ix) *
        cherenkov_track_length.(particle.energy, particle.type)
    )

    PhotonSource(
        particle.position,
        particle.direction,
        particle.time,
        total_contrib,
        CherenkovSpectrum,
        AngularEmissionProfile{:CherenkovAngularEmission})


end

function particle_to_elongated_lightsource!(
    particle::Particle{T},
    int_grid::AbstractArray{T},
    medium::MediumProperties,
    spectrum::Spectrum,
    wl_range::Tuple{T,T},
    output::Union{Zygote.Buffer,AbstractVector{PhotonSource}},
) where {T<:Real}


    """
    s = SobolSeq([range_cm[1]], [range_cm[2]])

    n_steps = Int64(ceil(ustrip(Unitful.NoUnits, (range[2] - range[1]) / precision)))
    int_grid = sort!(vec(reduce(hcat, next!(s) for i in 1:n_steps)))u"cm"
    """

    n_steps = length(int_grid)

    fractional_contrib_vec = Vector{T}(undef, n_steps)

    if typeof(output) <: Zygote.Buffer
        fractional_contrib = Zygote.Buffer(fractional_contrib_vec)
    else
        fractional_contrib = fractional_contrib_vec
    end

    fractional_contrib_long!(particle.energy, int_grid, medium, particle.type, fractional_contrib)

    total_contrib = (
        frank_tamm_norm(wl_range, wl -> get_refractive_index(wl, medium)) *
        cherenkov_track_length(particle.energy, particle.type)
    )


    step_along = [0.5 * (int_grid[i] + int_grid[i+1]) for i in 1:(n_steps-1)]

    for i in 2:n_steps
        this_pos = particle.position .+ step_along[i-1] .* particle.direction
        this_time = particle.time + step_along[i-1] / c_vac_m_p_ns

        this_nph = Int64(round(total_contrib * fractional_contrib[i]))

        this_src = PhotonSource(
            this_pos,
            particle.direction,
            this_time,
            this_nph,
            spectrum,
            AngularEmissionProfile{:CherenkovAngularEmission,Float64}())

        output[i-1] = this_src
    end

    output
end

function particle_to_elongated_lightsource(
    particle::Particle{T},
    len_range::Tuple{T,T},
    precision::T,
    medium::MediumProperties,
    spectrum::Spectrum,
    wl_range::Tuple{T,T}
) where {T<:Real}

    int_grid = range(len_range[1], len_range[2], step=precision)
    n_steps = size(int_grid, 1)
    output = Vector{PhotonSource}(undef, n_steps - 1)
    particle_to_elongated_lightsource!(particle, int_grid, medium, spectrum, wl_range, output)
end






end