include("src/photon_prob_cuda.jl")

using Unitful
using UnitfulRecipes
using Plots

using .Spectral
using .Medium
using .LightYield
using .Emission


log_energies = 2:0.1:8
zs = (0:10:3000.)u"cm"
medium = make_cascadia_medium_properties(Float64)

# Plot longitudinal profile
plot(zs, longitudinal_profile.(Ref(1E3u"GeV"), zs, Ref(medium), Ref(LightYield.EMinus)), label="1E3 GeV", title="Longitudinal Profile")
plot!(zs, longitudinal_profile.(Ref(1E5u"GeV"), zs, Ref(medium), Ref(LightYield.EMinus)), label="1E5 GeV")


# Show fractional contribution for a segment of shower depth
frac_contrib = fractional_contrib_long(1E5u"GeV", zs, medium, LightYield.EMinus)
plot(zs, frac_contrib, linetype=:steppost, label="", ylabel="Fractional light yield")
sum(frac_contrib) 

ftamm_norm = frank_tamm_norm((200u"nm", 800u"nm"), wl -> get_refractive_index(wl, medium))
light_yield = uconvert(Unitful.NoUnits, cherenkov_track_length.(1E5u"GeV", LightYield.EMinus) * ftamm_norm)

plot(zs, frac_contrib .*light_yield, linetype=:steppost, label="", ylabel="Light yield per segment")

# Calculate Cherenkov track length as function of energy 
tlens = cherenkov_track_length.((10 .^log_energies)u"GeV", LightYield.EMinus)
plot(log_energies, tlens, yscale=:log10, xlabel="Log10(E/GeV)", ylabel="Cherenkov track length")

total_lys = frank_tamm_norm((200u"nm", 800u"nm"), wl -> get_refractive_index(wl, medium)) * tlens

plot(log_energies, uconvert.(Unitful.NoUnits, total_lys), yscale=:log10, xlabel="Number of photons", ylabel="log10(Energy/GeV)")