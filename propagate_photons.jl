include("src/photon_prob_cuda.jl")

using CUDA
using Unitful
using UnitfulRecipes
using StaticArrays
using Plots
using StatsPlots 
using StatsBase
using QuadGK
using Statistics
using PhysicalConstants.CODATA2018
using Distributions
using Sobol
using DataFrames
using Parquet
using Random

using Printf
using Parameters: @with_kw
using .Spectral
using .Medium
using .LightYield
using .Emission
using .Detection

using Printf


log_energies = 2:0.1:8
zs = (0:100:3000.)u"cm"
medium = make_cascadia_medium_properties(Float64)


plot(zs, longitudinal_profile.(Ref(1E3u"GeV"), zs, Ref(medium), Ref(LongitudinalParametersEMinus)))
plot!(zs, longitudinal_profile.(Ref(1E5u"GeV"), zs, Ref(medium), Ref(LongitudinalParametersEMinus)))

frac_contrib = fractional_contrib_long(1E5u"GeV", zs, medium, LongitudinalParametersEMinus)

plot(zs, frac_contrib, linetype=:steppost)
sum(frac_contrib) 

tlens = cherenkov_track_length.((10 .^log_energies)u"GeV", Ref(CherenkovTrackLengthParametersEMinus))

plot(log_energies, tlens, yscale=:log10)


total_lys = frank_tamm_norm((200u"nm", 800u"nm"), wl -> get_refractive_index(wl, medium)) * tlens

plot(log_energies, uconvert.(Unitful.NoUnits, total_lys), yscale=:log10)


function get_dir_reweight(thetas, obs_angle, ref_ix)    
    norm = cherenkov_ang_dist_int.(ref_ix, Ref(-1.), Ref(1.)) .* 2
    cherenkov_ang_dist.(cos.(thetas .- obs_angle), ref_ix) ./ norm 
end





function fit_photon_dist(obs_angles, obs_photon_df, n_ph_gen)
    df = DataFrame(
        obs_angle=Float64[],
        fit_alpha=Float64[],
        fit_theta=Float64[],
        det_fraction=Float64[])

    ph_thetas = obs_photon_df[:, :theta]
    ph_ref_ix = obs_photon_df[:, :ref_ix]
    ph_abs_weight = obs_photon_df[:, :abs_weight]
    ph_tres = obs_photon_df[:, :tres]

    for obs_angle in obs_angles
        dir_weight = get_dir_reweight(ph_thetas, obs_angle, ph_ref_ix)
        total_weight = dir_weight .* ph_abs_weight
        
        mask = ph_tres .>= 0

        dfit = fit_mle(Gamma, ph_tres[mask], total_weight[mask])  
        push!(df, (obs_angle, dfit.α, dfit.θ, sum(total_weight) / n_ph_gen))
    end

    df
end

function make_photon_fits(n_photons_per_dist::Int64, n_distances::Integer, n_angles::Integer)

    s = SobolSeq([0.], [pi])
    medium = make_cascadia_medium_properties(Float32)

    s2 = SobolSeq([0f0], [Float32(log10(250))])
    distances = 10 .^ reduce(hcat, next!(s2) for i in 1:n_distances)
    results = map(dist -> propagate(dist, medium, n_photons_per_dist), distances)
    obs_angles = reduce(hcat, next!(s) for i in 1:n_angles)
    
    results = fit_photon_dist.(Ref(obs_angles), results, Ref(n_photons_per_dist))
    vcat(results..., source=:distance => vec(distances))
    
end
results_df = make_photon_fits(Int64(1E7), 100, 100)
write_parquet("photon_fits.parquet", results_df)









ix = 1

tres = results[ix][1]
wls = results[ix][2]
directions = results[ix][3]

map(size, [res[2] for res in results])











alphas = [fit.α for fit in fits]

plot(obs_angles, alphas)

weights
p = plot()
for (fit, w) in zip(fits, weights)

    xs = 1E-4:0.1:10
    vals = sum(w) .* pdf.(Ref(fit), xs)


    plot!(p, xs, vals)
end
p
plots = map(fit -> , fits)

plot(plots...)
p = histogram(tres[1], weights=total_weight, bins=bins=0.:0.1:10, normalize=true, yscale=:log10, ylim=(1E-4, 10))
  

                

