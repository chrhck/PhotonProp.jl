module 

include("src/photon_prob_cuda.jl")
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
using Base.Iterators




