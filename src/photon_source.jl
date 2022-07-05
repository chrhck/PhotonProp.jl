module PhotonSource

using ..Spectral

export EmissionProfile, Isotropic, Cherenkov
export PhotonSource

abstract type EmissionProfile end
struct Isotropic{T} <: EmissionProfile end
struct Cherenkov{T} <: EmissionProfile end

struct PhotonSource{T,U<:Spectrum,V<:EmissionProfile}
    position::SVector{3, T}
    direction::SVector{3, T}
    time::T
    photons::Int64
    spectrum::U
    emission_profile::V
end


end