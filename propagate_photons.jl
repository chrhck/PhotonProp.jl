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
using Flux
using Base.Iterators: partition

using Flux:params as fparams
using Flux.Data: DataLoader
using Flux.Losses: mse
using Flux:@epochs
using Printf
using Parameters: @with_kw
using .PhotonPropagation
using .Spectral
using .Medium
using .LightYield

using Printf



log_energies = 2:0.1:8
zs = (0:100:3000.)u"cm"
medium = make_cascadia_medium_properties(Float64)

plot(zs, longitudinal_profile.(Ref(1E3u"GeV"), zs, Ref(medium), Ref(LongitudinalParametersEMinus)))
plot!(zs, longitudinal_profile.(Ref(1E5u"GeV"), zs, Ref(medium), Ref(LongitudinalParametersEMinus)))

frac_contrib = fractional_contrib_long(1E5u"GeV", zs, MediumPropertiesWater, LongitudinalParametersEMinus)

plot(zs, frac_contrib, linetype=:steppost)
sum(frac_contrib) 

tlens = cherenkov_track_length.((10 .^log_energies)u"GeV", Ref(CherenkovTrackLengthParametersEMinus))

plot(log_energies, tlens, yscale=:log10)


total_lys = frank_tamm_norm((200u"nm", 800u"nm"), wl -> MediumPropertiesWater.ref_ix) * tlens

plot(log_energies, ustrip(total_lys), yscale=:log10)



plot(wls, cherenkov_counts.(wls, Ref(tlen[1]), Ref(MediumPropertiesWater)))



function get_dir_reweight(thetas, obs_angle, ref_ix)    
    norm = cherenkov_ang_dist_int.(ref_ix, Ref(-1.), Ref(1.)) .* 2
    cherenkov_ang_dist.(cos.(thetas .- obs_angle), ref_ix) ./ norm 
end


function propagate(distance::Float32, medium::MediumProperties, n_ph_gen::Int64)
   
    target_radius = 0.21f0
    source = PhotonSource(@SVector[0f0, 0f0, 0f0], 0f0, n_ph_gen, Cherenkov((300f0, 800f0), 20, medium), Isotropic{Float32}(), )
    target = PhotonTarget(@SVector[0f0, 0f0, distance], target_radius)

    threads = 1024
    blocks = 16

    stack_len = Int32(cld(1E5, blocks))

    positions, directions, wavelengths, dist_travelled, stack_idx, n_ph_sim = initialize_photon_arrays(stack_len, blocks, Float32)

    @cuda threads=threads blocks=blocks shmem=sizeof(Int32) cuda_propagate_photons!(
        positions, directions, wavelengths, dist_travelled, stack_idx, n_ph_sim, stack_len, Int32(0),
        Val(source), target.position, target.radius, Val(medium))
        
    distt = process_output(Vector(dist_travelled), Vector(stack_idx))
    wls = process_output(Vector(wavelengths), Vector(stack_idx))
    directions = process_output(Vector(directions), Vector(stack_idx))

    abs_weight =  convert(Vector{Float64}, exp.(- distt ./ get_absorption_length.(wls, Ref(medium))))

    ref_ix = get_refractive_index.(wls, Ref(medium))
    c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
    c_n = (c_vac ./ ref_ix)

    photon_times = distt ./ c_n    
    tgeo = (distance - target_radius) ./ (c_vac / get_refractive_index(800., medium))
    tres = (photon_times .- tgeo)
    thetas = map(dir -> acos(dir[3]), directions)
    
    DataFrame(tres=tres, theta=thetas, ref_ix=ref_ix, abs_weight=abs_weight)
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

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
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


@with_kw mutable struct Hyperparams
    batch_size::Int64
    learning_rate::Float64
    epochs::Int64
    width::Int64
end


function read_from_parquet(filename)
    results_df = DataFrame(read_parquet(filename))

    results_df[!,:] = convert.(Float32,results_df[!,:])
    results_df[!, :log_det_fraction] = -log10.(results_df[!, :det_fraction])
    results_df[!, :log_distance] = log10.(results_df[!, :distance])
    results_df[!, :cos_obs_angle] = cos.(results_df[!, :obs_angle])

    feature_names = [:log_distance, :cos_obs_angle]
    target_names = [:fit_alpha, :fit_theta, :log_det_fraction]

    df_train, df_test = splitdf(results_df, 0.8)


    features_train = Matrix{Float32}(df_train[:,feature_names])'
    targets_train = Matrix{Float32}(df_train[:, target_names])'
    features_test = Matrix{Float32}(df_test[:,feature_names])'
    targets_test = Matrix{Float32}(df_test[:, target_names])'

    return (features_train, targets_train, features_test, targets_test)


end

function get_data(args::Hyperparams)
    
    features_train, targets_train, features_test, targets_test = read_from_parquet("photon_fits.parquet")

    loader_train = DataLoader((features_train, targets_train), batchsize=args.batch_size, shuffle=true)
    loader_test = DataLoader((features_test, targets_test), batchsize=args.batch_size)

    loader_train, loader_test
end


function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += mse(model(x), y)
    end
    l/length(dataloader)
end


function train(; kws...)
    ## Initialize hyperparameter arguments
    args = Hyperparams(; kws...)

    ## Load processed data
    train_data, test_data = get_data(args)
    train_data = gpu.(train_data)
    test_data = gpu.(test_data)

    model = Chain(
        Dense(2, args.width, relu),
        Dense(args.width, args.width, relu),
        Dense(args.width, 3))
	

    model = gpu(model)
    loss(x, y) = mse(model(x), y)
	
    optimiser = ADAM(args.learning_rate)
    evalcb = () -> @show(loss_all(train_data, model))

    println("Starting training.")
    @epochs args.epochs Flux.train!(loss, fparams(model), train_data, optimiser, cb = evalcb)
	
    return model, test_data
end

model, test_data = train(epochs=20, width=300, learning_rate=0.005, batch_size=200)

@show loss_all(test_data, model)

data = read_from_parquet("photon_fits.parquet")
target_pred = cpu(model)(data[3])

feat_test = data[3]
targ_test = data[4]

l = @layout [a b c ; d e f]
plots = []
feature_names = [:log_distance, :cos_obs_angle]
target_names = [:fit_alpha, :fit_theta, :log_det_fraction]
for (i, j) in Base.product(1:3, 1:2)
    p = scatter(feat_test[j, :], targ_test[i, :], alpha=0.7, label="Truth",
        xlabel=feature_names[j], ylabel=target_names[i])
    scatter!(p, feat_test[j, :], target_pred[i, :], alpha=0.7, label="Prediction")
    push!(plots, p)
end
plot(plots..., layout=l)







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
  

                

