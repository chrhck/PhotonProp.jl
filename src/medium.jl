module Medium
using Unitful
using Base: @kwdef

using ..Utils

export make_cascadia_medium_properties
export salinity, pressure, temperature, vol_conc_small_part, vol_conc_large_part, radiation_length, density
export get_refractive_index, get_scattering_length, get_absorption_length
export MediumProperties, WaterProperties

@unit ppm "ppm" Partspermillion 1 // 1000000 false
Unitful.register(Medium)

abstract type MediumProperties{T} end


"""
    WaterProperties(salinity, presure, temperature, vol_conc_small_part, vol_conc_large_part)

Properties for a water-like medium. Use unitful constructor to create a value of this type.
"""
struct WaterProperties{T} <: MediumProperties{T}
    salinity::T # permille
    pressure::T # atm
    temperature::T #°C
    vol_conc_small_part::T # ppm
    vol_conc_large_part::T
    radiation_length::T # g / cm^2

    WaterProperties(::T, ::T, ::T, ::T, ::T, ::T) where {T} = error("Use unitful constructor")

    function WaterProperties(
        salinity::Unitful.Quantity{T},
        pressure::Unitful.Quantity{T},
        temperature::Unitful.Quantity{T},
        vol_conc_small_part::Unitful.Quantity{T},
        vol_conc_large_part::Unitful.Quantity{T},
        radiation_length::Unitful.Quantity{T}
    ) where {T}
        new{T}(
            ustrip(T, u"permille", salinity),
            ustrip(T, u"atm", pressure),
            ustrip(T, u"°C", temperature),
            ustrip(T, u"ppm", vol_conc_small_part),
            ustrip(T, u"ppm", vol_conc_large_part),
            ustrip(T, u"g/cm^2", radiation_length)
        )
    end
end

make_cascadia_medium_properties(T::Type) = WaterProperties(
    T(34.82)u"permille",
    T(269.44088)u"bar",
    T(1.8)u"°C",
    T(0.0075)u"ppm",
    T(0.0075)u"ppm",
    T(36.08)u"g/cm^2")



@kwdef struct DIPPR105Params
    A::Float64
    B::Float64
    C::Float64
    D::Float64
end

# http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=Water
DDBDIPR105Params = DIPPR105Params(A=0.14395, B=0.0112, C=649.727, D=0.05107)

"""
    DIPPR105(temperature::Real, params::DIPPR105Params=DDBDIPR105Params)

Use DPPIR105 formula to calculate water density as function of temperature.
temperature in K

Returns pressure in kg/m^3
"""
DIPPR105(temperature::Real, params::DIPPR105Params=DDBDIPR105Params) = params.A / params.B^(1 + (1 - temperature) / params.C)^params.D

salinity(::T) where {T<:MediumProperties} = error("Not implemented for $T")
salinity(x::WaterProperties) = x.salinity

pressure(::T) where {T<:MediumProperties} = error("Not implemented for $T")
pressure(x::WaterProperties) = x.pressure

temperature(::T) where {T<:MediumProperties} = error("Not implemented for $T")
temperature(x::WaterProperties) = x.temperature

density(::T) where {T<:MediumProperties} = error("Not implemented for $T")
density(x::WaterProperties) = DIPPR105(temperature(x))

vol_conc_small_part(::T) where {T<:MediumProperties} = error("Not implemented for $T")
vol_conc_small_part(x::WaterProperties) = x.vol_conc_small_part


vol_conc_large_part(::T) where {T<:MediumProperties} = error("Not implemented for $T")
vol_conc_large_part(x::WaterProperties) = x.vol_conc_large_part

radiation_length(::T) where {T<:MediumProperties} = error("Not implemented for $T")
radiation_length(x::WaterProperties) = x.radiation_length

"""
get_refractive_index_fry(wavelength, salinity, temperature, pressure)


    The phase refractive index of sea water according to a model
    from Quan&Fry. 

    wavelength is given in nm, salinity in permille, temperature in °C and pressure in atm
    
    The original model is taken from:
    X. Quan, E.S. Fry, Appl. Opt., 34, 18 (1995) 3477-3480.
    
    An additional term describing pressure dependence was included according to:
    Wolfgang H.W.A. Schuster, "Measurement of the Optical Properties of the Deep
    Mediterranean - the ANTARES Detector Medium.",
    PhD thesis (2002), St. Catherine's College, Oxford
    downloaded Jan 2011 from: http://www.physics.ox.ac.uk/Users/schuster/thesis0098mmjhuyynh/thesis.ps

    Adapted from clsim (©Claudio Kopper)
"""

function get_refractive_index_fry(
    wavelength::T;
    salinity::Real,
    temperature::Real,
    pressure::Real) where {T<:Real}

    n0 = 1.31405
    n1 = 1.45e-5
    n2 = 1.779e-4
    n3 = 1.05e-6
    n4 = 1.6e-8
    n5 = 2.02e-6
    n6 = 15.868
    n7 = 0.01155
    n8 = 0.00423
    n9 = 4382
    n10 = 1.1455e6

    a01 = (
        n0
        +
        (n2 - n3 * temperature + n4 * temperature^2) * salinity
        -
        n5 * temperature^2
        +
        n1 * pressure
    )
    a2 = n6 + n7 * salinity - n8 * temperature
    a3 = -n9
    a4 = n10

    x = 1 / wavelength
    return T(a01 + x * (a2 + x * (a3 + x * a4)))
end

function get_refractive_index_fry(
    wavelength::Unitful.Length{T};
    salinity::Unitful.DimensionlessQuantity,
    temperature::Unitful.Temperature,
    pressure::Unitful.Pressure) where {T<:Real}

    get_refractive_index_fry(
        ustrip(T, u"nm", wavelength),
        salinity=ustrip(u"permille", salinity),
        temperature=ustrip(u"°C", temperature),
        pressure=ustrip(u"atm", pressure)
    )
end

"""
    get_refractive_index(wavelength, medium)

    Return the refractive index at wavelength for medium
"""
get_refractive_index(wavelength::Real, medium::WaterProperties) = get_refractive_index_fry(
    wavelength,
    salinity=salinity(medium),
    temperature=temperature(medium),
    pressure=pressure(medium))

get_refractive_index(wavelength::Unitful.Length, medium::MediumProperties) = get_refractive_index(
    ustrip(u"nm", wavelength),
    medium)



"""
get_sca_len_part_conc(wavelength; vol_conc_small_part, vol_conc_large_part)

    Calculates the scattering length (in m) for a given wavelength based on concentrations of 
    small (`vol_conc_small_part`) and large (`vol_conc_large_part`) particles.
    wavelength is given in nm, vol_conc_small_part and vol_conc_large_part in ppm


    Adapted from clsim ©Claudio Kopper
"""
@inline function get_sca_len_part_conc(wavelength::T; vol_conc_small_part::Real, vol_conc_large_part::Real) where {T<:Real}

    ref_wlen::T = 550  # nm
    x = ref_wlen / wavelength

    sca_coeff = (
        0.0017 * x^4.3
        + 1.34 * vol_conc_small_part * x^1.7
        + 0.312 * vol_conc_large_part * x^0.3
    )

    return T(1 / sca_coeff)

end

function get_sca_len_part_conc(
    wavelength::Unitful.Length;
    vol_conc_small_part::Unitful.DimensionlessQuantity,
    vol_conc_large_part::Unitful.DimensionlessQuantity)

    get_sca_len_part_conc(
        ustrip(u"nm", wavelength),
        vol_conc_small_part=ustrip(u"ppm", vol_conc_small_part),
        vol_conc_large_part=ustrip(u"ppm", vol_conc_large_part))
end

"""
    get_scattering_length(wavelength, medium)

Return the scattering length for a given wavelength and medium
"""

@inline function get_scattering_length(wavelength::Real, medium::WaterProperties)
    get_sca_len_part_conc(wavelength, vol_conc_small_part=vol_conc_small_part(medium), vol_conc_large_part=vol_conc_large_part(medium))
end


function get_scattering_length(wavelength::Unitful.Length, medium::MediumProperties)
    get_scattering_length(ustrip(u"nm", wavelength), medium)
end


function get_absorption_length(wavelength::T, ::WaterProperties) where {T<:Real}

    x = [T(300.0), T(365.0), T(400.0), T(450.0), T(585.0), T(800.0)]
    y = [T(10.4), T(10.4), T(14.5), T(27.7), T(7.1), T(7.1)]

    fast_linear_interp(wavelength, x, y)


end


end # Module