using Revise
using StaticArrays
using Unitful
using Plots
using LinearAlgebra
includet("../medium.jl")
includet("../spectrum.jl")
includet("../emission.jl")
includet("../lightyield.jl")
includet("../detection.jl")

using .Medium
using .Spectral
using .Emission
using .LightYield
using .Detection


begin
    position = @SVector[0.0, 0.0, 0.0]
    direction = @SVector[0.0, 0.0, 1.0]
    energy = 1E3
    time = 0.0
    ptype = LightYield.EMinus
    p = Particle(position, direction, time, energy, ptype)
    medium = make_cascadia_medium_properties(Float64)

    sources = particle_to_elongated_lightsource(p, (0.0u"cm", 30.0u"m"), 0.5u"m", medium, (250.0u"nm", 800.0u"nm"))
end

positions = [source.position for source in sources]
photons = [source.photons for source in sources]
plot(photons)



