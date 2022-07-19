using Revise
includet("src/photon_prop.jl")

using Plots

using .PhotonPropagation
using .Spectral
using .Medium
using .LightYield
using .Emission


log_energies = 2:0.1:8
zs = (0:10:3000.0)# cm
medium = make_cascadia_medium_properties(Float64)

# Plot longitudinal profile
plot(zs, longitudinal_profile.(Ref(1E3), zs, Ref(medium), Ref(LightYield.EMinus)), label="1E3 GeV", title="Longitudinal Profile")
plot!(zs, longitudinal_profile.(Ref(1E5), zs, Ref(medium), Ref(LightYield.EMinus)), label="1E5 GeV")


# Show fractional contribution for a segment of shower depth
frac_contrib = fractional_contrib_long(1E5, zs, medium, LightYield.EMinus)
plot(zs, frac_contrib, linetype=:steppost, label="", ylabel="Fractional light yield")
sum(frac_contrib)

ftamm_norm = frank_tamm_norm((200, 800), wl -> get_refractive_index(wl, medium))
light_yield = cherenkov_track_length.(1E5, LightYield.EMinus)

plot(zs, frac_contrib .* light_yield, linetype=:steppost, label="", ylabel="Light yield per segment")

# Calculate Cherenkov track length as function of energy 
tlens = cherenkov_track_length.((10 .^ log_energies), LightYield.EMinus)
plot(log_energies, tlens, yscale=:log10, xlabel="Log10(E/GeV)", ylabel="Cherenkov track length")

total_lys = frank_tamm_norm((200.0, 800.0), wl -> get_refractive_index(wl, medium)) * tlens

plot(log_energies, total_lys, yscale=:log10, xlabel="Number of photons", ylabel="log10(Energy/GeV)")