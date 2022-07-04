using Revise
include("src/photon_prob_cuda.jl")

using CUDA
using Unitful
using StaticArrays
using Plots
using StatsPlots
using Statistics
using PhysicalConstants.CODATA2018
using Distributions
using Sobol
using DataFrames
using Parquet
using Random
using ScikitLearn
using ScikitLearn.GridSearch: RandomizedSearchCV
using PyCall: @pyimport

using .PhotonPropagation
using .Spectral
using .Medium


using Printf

@pyimport scipy.stats as stats

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

results_df = DataFrame(read_parquet("photon_fits.parquet"))
results_df[!, :log_det_fraction] = log10.(results_df[:, :det_fraction])

@df results_df marginalhist(:obs_angle, :log_det_fraction)

df_train, df_test = splitdf(results_df, 0.8)

feature_names = [:distance, :obs_angle]
target_names = [:fit_alpha, :fit_theta, :det_fraction]


features_train = df_train[:,feature_names]
targets_train = df_train[:, target_names]
features_test = df_test[:,feature_names]
targets_test = df_test[:, target_names]

mapper_features =  DataFrameMapper([(feature_names, nothing)])
mapper_targets = DataFrameMapper([(target_names, nothing)])

X = fit_transform!(mapper_features, copy(features_train))
Y = fit_transform!(mapper_targets, copy(targets_train))

X_test = fit_transform!(mapper_features, copy(features_test))
Y_test = fit_transform!(mapper_targets, copy(targets_test))

@sk_import ensemble: RandomForestRegressor

sampler(a, b) = stats.randint(a, b)
#sampler(a, b) = DiscreteUniform(a, b-1)  TODO

# specify parameters and distributions to sample from
param_dist = Dict("max_depth"=> [2, 3, 4, nothing],
                  "min_samples_split"=> sampler(2, 20),
                  "min_samples_leaf"=> sampler(1, 20),
                  "bootstrap"=> [true, false],
                  "criterion"=>["squared_error", "absolute_error"],
                  "n_estimators"=> sampler(100, 250))

clf = RandomForestRegressor(random_state=2, max_features=1)

n_iter_search = 50
random_search = RandomizedSearchCV(
    clf, param_dist, n_iter=n_iter_search, random_state=MersenneTwister(42),
  
    )
fit!(random_search, X, Y)


random_search_dist = RandomizedSearchCV(
    clf, param_dist, n_iter=n_iter_search, random_state=MersenneTwister(42),
  
    )
fit!(random_search_dist, X, Y[:, 1:2])


function report(grid_scores, n_top=5)
    top_scores = sort(grid_scores, by=x->x.mean_validation_score, rev=true)[1:n_top]
    for (i, score) in enumerate(top_scores)
        println("Model with rank:$i")
        @printf("Mean validation score: %.3f (std: %.3f)\n",
                score.mean_validation_score,
                std(score.cv_validation_scores))
        println("Parameters: $(score.parameters)")
        println("")
    end
end

report(random_search.grid_scores_)

best_params = random_search.best_params_

#clf = RandomForestRegressor(random_state=2, max_features=1)


Y_pred = predict(random_search, X_test)

score(random_search, X_test, Y_test)

l = @layout [a b c ; d e f]
plots = []
for (i, j) in Base.product(1:3, 1:2)
    p = scatter(X_test[:, j], Y_test[:, i], alpha=0.7, label="Truth",
        xlabel=feature_names[j], ylabel=target_names[i])
    scatter!(p, X_test[:, j], Y_pred[:, i], alpha=0.7, label="Prediction")
    push!(plots, p)
end
plot(plots..., layout=l)


#@show score(random_search, X, Y)


names(results_df)

#histogram(tres, bins=0.:1:20.)
#histogram(tres[1], bins=0.:0.1:10, yscale=:log10)
#plot(distances, size.(tres, Ref(1)))

rand(1:2, 10)

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
  

                

