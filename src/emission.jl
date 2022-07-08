module Emission

using StaticArrays
using SpecialFunctions
using Interpolations

using ..Spectral

export AngularEmissionProfile
export PhotonSource
export cherenkov_ang_dist, cherenkov_ang_dist_int

struct AngularEmissionProfile{U,T} end

struct CherenkovAngDistParameters{T<:Real}
    a::T
    b::T
    c::T
    d::T
end

# params for e-
STD_ANG_DIST_PARS = CherenkovAngDistParameters(4.27033, -6.02527, 0.29887, -0.00103)

"""
cherenkov_ang_dist(costheta, ref_index)

    Angular distribution of cherenkov photons for EM cascades.

    Taken from https://arxiv.org/pdf/1210.5140.pdf
"""
function cherenkov_ang_dist(
    costheta::Real,
    ref_index::Real,
    ang_dist_pars::CherenkovAngDistParameters=STD_ANG_DIST_PARS)

    cos_theta_c = 1 / ref_index
    a = ang_dist_pars.a
    b = ang_dist_pars.b
    c = ang_dist_pars.c
    d = ang_dist_pars.d

    return a * exp(b * abs(costheta - cos_theta_c)^c) + d
end


"""
    cherenkov_ang_dist_int(ref_index, lower, upper, ang_dist_pars)

Integral of the cherenkov angular distribution function.
"""

function _cherenkov_ang_dist_int(
    ref_index::Real,
    lower::Real=-1.0,
    upper::Real=1,
    ang_dist_pars::CherenkovAngDistParameters=STD_ANG_DIST_PARS)

    a = ang_dist_pars.a
    b = ang_dist_pars.b
    c = ang_dist_pars.c
    d = ang_dist_pars.d

    cos_theta_c = 1.0 / ref_index

    function indef_int(x)

        function lower_branch(x, cos_theta_c)
            return (1 / c * (c * d * x + (a * (cos_theta_c - x) * gamma(1 / c, -(b * (cos_theta_c - x)^c))) * (-(b * (cos_theta_c - x)^c))^(-1 / c)))
        end

        function upper_branch(x, cos_theta_c)
            return (1 / c * (c * d * x + (a * (cos_theta_c - x) * gamma(1 / c, -(b * (-cos_theta_c + x)^c))) * (-(b * (-cos_theta_c + x)^c))^(-1 / c)))
        end

        peak_val = lower_branch(cos_theta_c - 1e-5, cos_theta_c)

        if x <= cos_theta_c
            return lower_branch(x, cos_theta_c)
        else
            return upper_branch(x, cos_theta_c) + 2 * peak_val
        end
    end

    return indef_int(upper) - indef_int(lower)
end

struct ChAngDistInt
    interpolation
end

function interp_ch_ang_dist_int()
    ref_ixs = 1.1:0.01:1.5
    A = map(rfx -> _cherenkov_ang_dist_int(rfx, -1, 1), ref_ixs)
    ChAngDistInt(LinearInterpolation(ref_ixs, A))
end

(f::ChAngDistInt)(ref_ix::Real) = f.interpolation(ref_ix)
cherenkov_ang_dist_int = interp_ch_ang_dist_int()


struct PhotonSource{T,U<:Spectrum,V<:AngularEmissionProfile}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    photons::Int64
    spectrum::U
    emission_profile::V
end


end