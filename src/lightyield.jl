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
using Unitful
using Sobol
using PhysicalConstants.CODATA2018
using ..Emission
using ..Spectral
using ..Medium


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
    alpha::typeof(1.0u"cm")
    beta::Float64
    alpha_dev::typeof(1.0u"cm")
    beta_dev::Float64
end

CherenkovTrackLengthParametersEMinus = CherenkovTrackLengthParameters(
    alpha=532.07078881u"cm",
    beta=1.00000211,
    alpha_dev=5.78170887u"cm",
    beta_dev=0.5
)

CherenkovTrackLengthParametersEPlus = CherenkovTrackLengthParameters(
    alpha=532.11320598u"cm",
    beta=0.99999254,
    alpha_dev=5.73419669u"cm",
    beta_dev=0.5
)

CherenkovTrackLengthParametersGamma = CherenkovTrackLengthParameters(
    alpha=532.08540905u"cm",
    beta=0.99999877,
    alpha_dev=5.78170887u"cm",
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
    energy::Unitful.Energy,
    long_pars::LongitudinalParameters
)
    long_pars.alpha + long_pars.beta * log10(ustrip(u"GeV", energy))
end
long_parameter_a_edep(energy::Unitful.Energy, ptype::ParticleType) = long_parameter_a_edep(energy, get_longitudinal_params(ptype))

long_parameter_b_edep(::Unitful.Energy, long_pars::LongitudinalParameters) = long_pars.b
long_parameter_b_edep(energy::Unitful.Energy, ptype::ParticleType) = long_parameter_b_edep(energy, get_longitudinal_params(ptype))

function longitudinal_profile(
    energy::Unitful.Energy{T}, z::Unitful.Length{T}, medium::MediumProperties, long_pars::LongitudinalParameters) where {T<:Real}
    lrad = radiation_length(medium)u"g/cm^2" / density(medium)u"kg/m^3"
    t = uconvert(Unitful.NoUnits, z / lrad)
    b = long_parameter_b_edep(energy, long_pars)
    a = long_parameter_a_edep(energy, long_pars)

    b * ((t * b)^(a - 1.0) * exp(-(t * b)) / gamma(a))
end

function longitudinal_profile(
    energy::Unitful.Energy{T}, z::Unitful.Length{T}, medium::MediumProperties, ptype::ParticleType) where {T<:Real}
    longitudinal_profile(energy, z, medium, get_longitudinal_params(ptype))
end

function fractional_contrib_long(
    energy::Unitful.Energy,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    long_pars::LongitudinalParameters,
) where {T<:Unitful.Length}

    int_range = extrema(z_grid)
    norm = quadgk(
        z -> longitudinal_profile(energy, z, medium, long_pars),
        int_range[1], int_range[2])[1]

    part_contribs = Vector{Float64}(undef, size(z_grid, 1))
    part_contribs[1] = 0
    @inbounds for i in 1:size(z_grid, 1)-1
        part_contribs[i+1] = (
            1 / norm *
            quadgk(
                z -> longitudinal_profile(energy, z, medium, long_pars),
                z_grid[i], z_grid[i+1])[1]
        )
    end
    part_contribs
end

function fractional_contrib_long(
    energy::Unitful.Energy,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    ptype::ParticleType,
) where {T<:Unitful.Length}
    fractional_contrib_long(energy, z_grid, medium, get_longitudinal_params(ptype))
end



function cherenkov_track_length_dev(energy::Unitful.Energy, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha_dev * ustrip(u"GeV", energy)^track_len_params.beta_dev
end
cherenkov_track_length_dev(energy::Unitful.Energy, ptype::ParticleType) = cherenkov_track_length_dev(energy, get_track_length_params(ptype))

function cherenkov_track_length(energy::Unitful.Energy, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha * ustrip(u"GeV", energy)^track_len_params.beta
end
cherenkov_track_length(energy::Unitful.Energy, ptype::ParticleType) = cherenkov_track_length(energy, get_track_length_params(ptype))

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
    wl_range::Tuple{Unitful.Length{T},Unitful.Length{T}}
) where {T<:Real}

    total_contrib = (
        frank_tamm_norm(wl_range, wl -> medium.ref_ix) *
        cherenkov_track_length.(particle.energy * 1u"GeV", particle.type)
    )


    PhotonSource(
        particle.position,
        particle.direction,
        particle.time,
        total_contrib,
        CherenkovSpectrum,
        AngularEmissionProfile{:CherenkovAngularEmission})


end

function particle_to_elongated_lightsource(
    particle::Particle{T},
    range::Tuple{Unitful.Length{T},Unitful.Length{T}},
    precision::Unitful.Length{T},
    medium::MediumProperties,
    wl_range::Tuple{Unitful.Length{T},Unitful.Length{T}}
) where {T<:Real}


    range_cm = (ustrip(u"cm", range[1]), ustrip(u"cm", range[2]))
    s = SobolSeq([range_cm[1]], [range_cm[2]])

    n_steps = Int64(ceil(ustrip(Unitful.NoUnits, (range[2] - range[1]) / precision)))
    int_grid = sort!(vec(reduce(hcat, next!(s) for i in 1:n_steps)))u"cm"

    fractional_contrib = fractional_contrib_long(particle.energy * 1u"GeV", int_grid, medium, particle.type)
    total_contrib = (
        frank_tamm_norm(wl_range, wl -> get_refractive_index(wl, medium)) *
        cherenkov_track_length(particle.energy * 1u"GeV", particle.type)
    )

    sources = Vector{PhotonSource{T}}(undef, n_steps - 1)
    spectrum = CherenkovSpectrum(wl_range, 20, medium)

    pos_along = 0.5 .* (int_grid[1:end-1] .+ int_grid[2:end])

    for i in 2:n_steps
        step_along = ustrip(u"m", pos_along[i-1])

        this_pos = particle.position .+ step_along .* particle.direction
        this_time = particle.time + ustrip(u"ns", step_along * 1u"m" / SpeedOfLightInVacuum)

        this_nph = Int64(round(ustrip(Unitful.NoUnits, total_contrib * fractional_contrib[i])))

        this_src = PhotonSource(
            this_pos,
            particle.direction,
            this_time,
            this_nph,
            spectrum,
            AngularEmissionProfile{:CherenkovAngularEmission,Float64}())

        sources[i-1] = this_src
    end

    sources
end






end