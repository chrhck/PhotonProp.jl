using Plots
using StatsPlots
using Parquet
using Hyperopt

include("src/photon_prop.jl")

using .PhotonPropagation

infile = "PhotonProp.jl/photon_fits.parquet"

ho = @hyperopt for i = 50,
    sampler = RandomSampler(), # This is default if none provided
    epochs = 100:1000,
    batch_size = [300, 400, 500],
    dropout_rate = LinRange(0.1, 0.9, 1000),
    width = 100:100:700,
    learning_rate = 10 .^ LinRange(-4, -2, 1000)

    model, test_data = Modelling.train_mlp(
        epochs=epochs, width=width, learning_rate=learning_rate, batch_size=batch_size, data_file=infile,
        dropout_rate=dropout_rate)
    print("Epochs: $epochs, Width: $width, lr: $learning_rate, batch: $batch_size, dropout: $dropout_rate ")
    @show Modelling.loss_all(test_data, model)
end

plot(ho)

