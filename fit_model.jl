using Plots
using StatsPlots
using Parquet
using Hyperopt
using Random

include("src/photon_prop.jl")

using .PhotonPropagation

infile = joinpath(@__DIR__, "photon_fits.parquet")
rng = MersenneTwister(31338)
ho = @hyperopt for i = 10,
    sampler = RandomSampler(), # This is default if none provided
    epochs = 300,
    batch_size = [1024, 2048, 4096, 8192],
    dropout_rate = LinRange(0, 0.5, 1000),
    width = 128:64:1024,
    learning_rate = 10 .^ LinRange(-4, -2, 1000)

    model, test_data = Modelling.train_mlp(
        epochs=epochs, width=width, learning_rate=learning_rate, batch_size=batch_size, data_file=infile,
        dropout_rate=dropout_rate, rng=rng)
    print("Epochs: $epochs, Width: $width, lr: $learning_rate, batch: $batch_size, dropout: $dropout_rate ")
    @show Modelling.loss_all(test_data, model)
end

plot(ho, yscale=:log10)

