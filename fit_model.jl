includet("src/photon_prob_cuda.jl")


using .PhotonPropagation
using .Medium
using .LightYield
using .Emission
using .Detection

using Flux
using CUDA
using Base.Iterators: partition
using Flux: params as fparams
using Flux.Data: DataLoader
using Flux.Losses: mse
using Flux: @epochs
using Parameters: @with_kw
using Parquet
using DataFrames
using Distributions
using Plots
using Random
using Unitful
using StaticArrays
using StatsPlots
using LinearAlgebra




names(PhotonPropagation)

@with_kw mutable struct Hyperparams
    batch_size::Int64
    learning_rate::Float64
    epochs::Int64
    width::Int64
end

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end


function read_from_parquet(filename)
    results_df = DataFrame(read_parquet(filename))

    results_df[!, :] = convert.(Float32, results_df[!, :])
    results_df[!, :log_det_fraction] = -log10.(results_df[!, :det_fraction])
    results_df[!, :log_distance] = log10.(results_df[!, :distance])
    results_df[!, :cos_obs_angle] = cos.(results_df[!, :obs_angle])

    feature_names = [:log_distance, :cos_obs_angle]
    target_names = [:fit_alpha, :fit_theta, :log_det_fraction]

    df_train, df_test = splitdf(results_df, 0.8)

    features_train = Matrix{Float32}(df_train[:, feature_names])'
    targets_train = Matrix{Float32}(df_train[:, target_names])'
    features_test = Matrix{Float32}(df_test[:, feature_names])'
    targets_test = Matrix{Float32}(df_test[:, target_names])'

    return (features_train, targets_train, features_test, targets_test)
end
function get_data(args::Hyperparams)

    features_train, targets_train, features_test, targets_test = read_from_parquet("/home/chrhck/repos/julia/PhotonProp.jl/photon_fits.parquet")

    loader_train = DataLoader((features_train, targets_train), batchsize=args.batch_size, shuffle=true)
    loader_test = DataLoader((features_test, targets_test), batchsize=args.batch_size)

    loader_train, loader_test
end

function loss_all(dataloader, model)
    l = 0.0f0
    for (x, y) in dataloader
        l += mse(model(x), y)
    end
    l / length(dataloader)
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
    evalcb = () -> nothing # @show(loss_all(train_data, model))

    println("Starting training.")
    @epochs args.epochs Flux.train!(loss, fparams(model), train_data, optimiser, cb=evalcb)

    return model, test_data
end

function source_to_input(source::PhotonSource, target::PhotonTarget)
    em_rec_vec = source.position .- target.position
    distance = norm(em_rec_vec)
    em_rec_vec = em_rec_vec ./ distance
    cos_obs_angle = dot(em_rec_vec, source.direction)

    return log10(distance), cos_obs_angle

end



model, test_data = train(epochs=20, width=300, learning_rate=0.005, batch_size=200)

@show loss_all(test_data, model)

data = read_from_parquet("/home/chrhck/repos/julia/PhotonProp.jl/photon_fits.parquet")
target_pred = cpu(model)(data[3])

feat_test = data[3]
targ_test = data[4]

l = @layout [a b c; d e f]
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
target_pos = @SVector[100.0, 10.0, 0.0]
target = DetectionSphere(target_pos, 0.21)



model_input = Matrix{Float32}(undef, (2, 1))
model_input[:, 1] .= source_to_input(sources[1], target)

model_pred = cpu(model)(model_input)




em_rec_vec = sources[1].position .- target.position
distance = norm(em_rec_vec)

medium = make_cascadia_medium_properties(Float32)



prop_result = propagate_distance(Float32(distance), medium, 100000)#

histogram(prop_result[:, :tres], normalize=true)
plot!(Gamma(model_pred[1], model_pred[2]))