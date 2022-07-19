### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 3e649cbf-1546-4204-82de-f6db5be401c7
begin
    import Pkg
    Pkg.activate(".")

	using PlutoUI
	using Plots
	using StatsPlots
	using Parquet
	using StaticArrays
	using Unitful
	using LinearAlgebra
	using Distributions
	using Base.Iterators
	using Hyperopt
	using Random
	using Interpolations
	
	#ingredients (generic function with 1 method)
	function ingredients(path::String)
	    # this is from the Julia source code (evalfile in base/loading.jl)
	    # but with the modification that it returns the module instead of the last object
	    name = Symbol(basename(path))
	    m = Module(name)
	    Core.eval(m,
	        Expr(:toplevel,
	             :(eval(x) = $(Expr(:core, :eval))($name, x)),
	             :(include(x) = $(Expr(:top, :include))($name, x)),
	             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
	             :(include($path))))
	    m
	end

	PhotonPropagation = ingredients("src/photon_prop.jl").PhotonPropagation
	Medium = PhotonPropagation.Medium
	Emission = PhotonPropagation.Emission
	Detection = PhotonPropagation.Detection
	LightYield = PhotonPropagation.LightYield
	Modelling = PhotonPropagation.Modelling
	Spectral = PhotonPropagation.Spectral

	using Flux
	using BSON: @save, @load
	using BSON
end

# ╔═╡ db772593-2e58-4cec-bc88-7113ac028811
using Zygote

# ╔═╡ 5ab60c75-fd19-4456-9121-fb42ce3e086f
md"""
# Photon detection model for EM cascade segments
"""

# ╔═╡ a615b299-3868-4083-9db3-806d7390eca1
md"""
## Create a medium and propagate photons
Propagate photons from an isotropic Cherenkov emitter to a spherical detector.

Distance: $(@bind distance Slider(5f0:0.1f0:200f0, default=25, show_value=true)) m

"""


# ╔═╡ 4db5a2bd-97bc-412c-a1d3-cb8391425e20
begin
	n_photons = 100000
	medium = Medium.make_cascadia_medium_properties(Float32)
	prop_result = PhotonPropagation.propagate_distance(distance, medium, n_photons)
	nothing
end

# ╔═╡ b37de33c-af67-4983-9932-2c32e8db399f
@df prop_result corrplot(cols(1:5))

# ╔═╡ 911c5c61-bcaa-4ba4-b5c0-09a112b6e877
md"""
The resulting DataFrame contains information about the detected photons:
- time-residual (`tres`)
- initial emission angle (`initial_theta`)
- refractive index for the photon's wavelength (`ref_ix`) 
- absorption weight (`abs_weight`)

The time-residual is the photon arrival time relative to the geometric path length 
```math
\begin{align}
t_{res} = t - t_{geo}(800 \,\mathrm{nm}) \\
t_{geo}(\lambda) = \frac{\mathrm{distance}}{\frac{c_{vac}}{n(\lambda)}}
\end{align}
```

Note: The detector is assumed to be 100% efficient. Lower detection efficiences can be trivialled added by an additional (wl-dependent) weight.

Photons can be reweighted to a Cherenkov angular emission spectrum:

"""




# ╔═╡ 30c4a297-715d-4268-b91f-22d0cac66511
begin
	my_obs_angle = deg2rad(70)
	dir_reweight = Modelling.get_dir_reweight(
		prop_result[:, :initial_theta],
		my_obs_angle,
		prop_result[:, :ref_ix])
	total_weight = dir_reweight .* prop_result[:, :abs_weight]
	histogram(prop_result[:, :tres], weights=total_weight, label="", xlabel="Time Residual (ns)", ylabel="Detected Photons")
end

# ╔═╡ 26469e97-51a8-4c00-a69e-fe50ad3a625a
md"""
## Fit Distributions
Runs the photon propagation for multiple distances and reweights those simulations to multiple observation angles. Fits the resulting arrival time distributions with Gamma-PDFs.
"""

# ╔═╡ 74b11928-fe9c-11ec-1d37-01f9b1e48fbe
begin	
	#results_df = Modelling.make_photon_fits(Int64(1E8), 250, 250, 300f0)
	#write_parquet("photon_fits.parquet", results_df)
	results_df = read_parquet("photon_fits.parquet")
end#


# ╔═╡ 26c7461c-5475-4aa9-b87e-7a55b82cea1f
@df results_df corrplot(cols(1:5))

# ╔═╡ 78695b21-e992-4f5c-8f34-28c3027e3179
@df results_df scatter(:distance, :det_fraction, yscale=:log10)

# ╔═╡ b9af02cb-b675-4030-b3c8-be866e85ebc7
md"""
## Fit distribution parameters with MLP
Here we fit a simple MLP to predict the distribution parameters (and the photon survival rate) as function of the distance and the observation angle
"""

# ╔═╡ cb97110d-97b2-410d-8ce4-bef9accfcef2
begin

	rng = MersenneTwister(31338)
	params = Dict(
		:width=>1024,
		:learning_rate=>0.0009,
		:batch_size=>1024,
		:data_file=>"photon_fits.parquet",
		:dropout_rate=>0.5,
		:rng=>rng,
		:epochs=>400
		)
	
	
	# Parameters from hyperparam optimization

	#=
	model, train_data, test_data, trafos = Modelling.train_mlp(;params...)
	@show Modelling.loss_all(test_data, model)

	model = cpu(model)
	@save "photon_model.bson" model
	=#
	model = BSON.load("photon_model.bson", @__MODULE__)[:model]
	#@load "photon_model.bson" model
	
	model = gpu(model)
	train_data, test_data, trafos = Modelling.get_data(Modelling.Hyperparams(;params...))

	train_data = gpu.(train_data)
	test_data = gpu.(test_data)
		
	output_trafos = [
		trafos[(:fit_alpha, :log_fit_alpha)],
		trafos[(:fit_theta, :log_fit_theta)],
		trafos[(:det_fraction, :log_det_fraction_scaled)]
	]
	

end

# ╔═╡ 8638dde4-9f02-4dbf-9afb-32982390a0b6
begin
	feat_test = hcat(collect(td[1] for td in test_data)...) 
	targ_test = hcat(collect(td[2] for td in test_data)...) |> cpu
	target_pred_test = model(feat_test) |> cpu
	feat_test = cpu(feat_test)

	feat_train = hcat(collect(td[1] for td in train_data)...) 
	targ_train = hcat(collect(td[2] for td in train_data)...) |> cpu
	target_pred_train = model(feat_train) |> cpu
	feat_train = cpu(feat_train)
	
	
	l = @layout [a b c; d e f]
	plots = []
	feature_names = [:log_distance, :cos_obs_angle]
	target_names = [:log_fit_alpha, :log_fit_theta, :log_det_fraction]
	
	for (i, j) in Base.product(1:3, 1:2)
	    p = scatter(feat_test[j, :], targ_test[i, :], alpha=0.7, label="Truth",
	        xlabel=feature_names[j], ylabel=target_names[i], ylim=(-1, 2), legend=:topleft)
	    scatter!(p, feat_test[j, :], target_pred_test[i, :], alpha=0.7, label="Prediction")
	    push!(plots, p)
	end

	plot(plots..., layout=l)
	
end

# ╔═╡ 984b11c9-2cd5-4dd6-9f36-26eec69eb17d
begin
	histogram(targ_train[3, :] - target_pred_train[3, :], normalize=true)
	histogram!(targ_test[3, :] - target_pred_test[3, :], normalize=true)
	
	
end

# ╔═╡ 0a37ce94-a949-4c3d-9cd7-1a64b1a3ce47
md"""
## Compare model prediction to MC
"""

# ╔═╡ be1a6f87-65e3-4741-8b98-bb4d305bd8c3
begin
    position = @SVector[0.0, 0.0, 0.0]
    direction = @SVector[0.0, 0.0, 1.0]
    energy = 1E5
    time = 0.0
	photons = 1000000
	medium64 = Medium.make_cascadia_medium_properties(Float64)
	spectrum = Spectral.CherenkovSpectrum((300.0, 800.0), 20, medium64)
	em_profile = Emission.AngularEmissionProfile{:IsotropicEmission, Float64}()
	
	source = Emission.PhotonSource(position, direction, time, photons, spectrum, em_profile)
	

	target_pos = @SVector[-10., 0.0, 10.0]
	target = Detection.DetectionSphere(target_pos, 0.21)
	
	model_input = Matrix{Float32}(undef, (2, 1))
	model_input[:, 1] .= Modelling.source_to_input(source, target)
	model_pred = cpu(model(gpu(model_input)))

	Modelling.transform_model_output!(model_pred, output_trafos)
	
	n_photons_pred = photons * model_pred[3, 1]

	
	this_prop_result = PhotonPropagation.propagate_distance(Float32(exp10(model_input[1, 1])), medium, photons)

	this_total_weight = (
		Modelling.get_dir_reweight(this_prop_result[:, :initial_theta],
			acos(model_input[2, 1]), this_prop_result[:, :ref_ix])
		.* this_prop_result[:, :abs_weight]
		.* Detection.p_one_pmt_acc.(this_prop_result[:, :wavelength])
	)
	
	histogram(this_prop_result[:, :tres], weights=this_total_weight, normalize=false,
	xlabel="Time (ns)", ylabel="# Photons", label="MC")

	xs_plot = 0:0.1:15
	
	gamma_pdf = Gamma(model_pred[1], model_pred[2])
	
	plot!(xs_plot, n_photons_pred .* pdf.(gamma_pdf, xs_plot), label="Model")
	#n_photons_pred, sum(this_total_weight)
end





# ╔═╡ 57450082-04dc-4d1a-8ef4-e321c3971c84
n_photons_pred, sum(this_total_weight)

# ╔═╡ 31ab7848-b1d8-4380-872e-8a12c3331051
md"""
## Use model to simulate an event for a detector.

The "particle" (EM-cascade) is split into a series of point-like Cherenov emitters, aligned along the cascade axis. Their respective lightyield is calculated from the longitudinal profile of EM cascades.
"""

# ╔═╡ 22e419a4-a4f9-48d5-950b-8420854c475a
begin
	Emission.frank_tamm_norm((300., 800.), wl -> Medium.get_refractive_index(wl, medium)) * LightYield.cherenkov_track_length(1E5, LightYield.EMinus)	
	
end

# ╔═╡ 4f7a9060-33dc-4bb8-9e41-eae5b9a30aa6
begin
	function make_detector_cube(nx, ny, nz, spacing_vert, spacing_hor)
	
		positions = Vector{SVector{3, Float64}}(undef, nx*ny*nz)
	
		lin_ix = LinearIndices((1:nx, 1:ny, 1:nz))
		for (i, j, k) in product(1:nx, 1:ny, 1:nz)
			ix = lin_ix[i, j, k]
			positions[ix] = @SVector [-0.5*spacing_hor*nx + (i-1)*spacing_hor, -0.5*spacing_hor*ny + (j-1)*spacing_hor, -0.5*spacing_vert*nz + (k-1)*spacing_vert]
		end
	
		positions
	
	end
	
	function make_targets(positions)
		targets = map(pos -> Detection.DetectionSphere(pos, 0.21), positions)
	end
		
end

# ╔═╡ 40940de0-c45b-4236-82f0-54a77d5fbb9a
begin
	positions = make_detector_cube(5, 5, 10, 50., 100.)
	
	xs = [p[1] for p in positions]
	ys = [p[2] for p in positions]
	
	scatter(xs, ys)
end

# ╔═╡ 709a7e6f-1e54-4350-b826-66b9097cc46a
begin
	function evaluate_model(
		targets::AbstractVector{U},
		particle::LightYield.Particle,
		medium::Medium.MediumProperties )  where {U <: Detection.PhotonTarget}

		sources = LightYield.particle_to_elongated_lightsource(
			particle,
			(0.0, 3000.0),
			10.,
			medium,
			spectrum,
			(300.0, 800.0),
			)
		
		inputs = Modelling.source_to_input(sources, targets)
		predictions = cpu(model(gpu(inputs)))
		Modelling.transform_model_output!(predictions, output_trafos)
		predictions = reshape(predictions, (3, size(sources, 1), size(targets, 1)))

		return predictions, sources
	end
	
	function shape_mixture_per_module(
		params::AbstractArray{U, 3},
	) where {U <: Real}

		n_sources = size(params, 2)
		n_targets = size(params, 3)
		T = MixtureModel{Univariate, Distributions.Continuous, Gamma{U}, Distributions.Categorical{U, Vector{U}}}
		mixtures = Vector{T}(undef, n_targets)
		for i in 1:n_targets

			dists = [Gamma(params[1, j, i], params[2, j, i]) for j in 1:n_sources]
			probs = params[3, :, i] ./ sum(params[3, :, i])
			mixtures[i] = MixtureModel(dists, probs)
		end
		mixtures
	end

	function shape_mixture_per_module(
		targets::AbstractVector{U},
		particle::LightYield.Particle,
		medium::Medium.MediumProperties) where {U <: Detection.PhotonTarget}

		predictions, _ = evaluate_model(targets, particle, medium)
		shape_mixture_per_module(predictions)
	end
		

	function poisson_dist_per_module(
		params::AbstractArray{U, 3},
		sources::AbstractVector{V}) where {U <: Real, V <: Emission.PhotonSource}
		pred = vec(sum(params[3, :, :] .* [src.photons for src in sources], dims=1))
		Poisson.(pred)
	end

	function poisson_dist_per_module(
		targets::AbstractVector{U},
		particle::LightYield.Particle,
		medium::Medium.MediumProperties) where {U <: Detection.PhotonTarget}

		predictions, sources = evaluate_model(targets, particle, medium)
		poisson_dist_per_module(predictions, sources)
	end
	

	function extended_llh_per_module(
		x::AbstractVector{T},
		poisson::Sampleable,
		shape::Sampleable) where {T <: Real}
		n_obs = size(x, 1)

		pllh = loglikelihood(poisson, n_obs)
		sllh = loglikelihood.(shape, x)

		return pllh + sum(sllh)
	
	end

	function extended_llh_per_module(
		x::AbstractVector{U},
		targets::AbstractVector{V},
		particle::LightYield.Particle,
		medium::Medium.MediumProperties) where {T <: Real, V <: Detection.PhotonTarget, U <: AbstractVector{T}}


		predictions, _ = evaluate_model(targets, particle, medium)
		shape = shape_mixture_per_module(predictions)
		poisson = poisson_dist_per_module(predictions, sources)


		extended_llh_per_module.(x, poisson, shape)
		
	end


	function sample_event(
		poissons::AbstractVector{U},
		shapes::AbstractVector{V}) where {U <: Sampleable, V <: Sampleable}

		if size(poissons) != size(shapes)
			error("Distribution vectors have to be of same size")
		end
		
		event = Vector{Vector{Float64}}(undef, size(poissons))
		for i in 1:size(poissons, 1)
			n_ph = rand(poissons[i])
			event[i] = rand(shapes[i], n_ph)
		end
		
		event
	end
	
end





# ╔═╡ 820e698c-f1c4-4d07-90d8-a4a13ff9cccd
md"""
### Calculate likelihoods

Particle zenith: $(@bind zenith_angle Slider(0:0.1:180, default=25, show_value=true)) deg

Particle azimuth $(@bind azimuth_angle Slider(0:0.1:360, default=180, show_value=true)) deg

"""

# ╔═╡ 462567a7-373f-4ac6-a11a-4c6853a8c45a

begin

	function sph_to_cart(theta::T, phi::T) where {T <:Real}
		sin_theta, cos_theta = sincos(theta)
    	sin_phi, cos_phi = sincos(phi)

	    x::T = cos_phi * sin_theta
	    y::T = sin_phi * sin_theta
	    z::T = cos_theta

		return SA[x, y, z]
	end

	targets = make_targets(positions)
	particle = LightYield.Particle(
		@SVector[0., 0., 20.],
	    sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
	    0.,
	    1E5,
	    LightYield.EMinus
	)

	poissons = poisson_dist_per_module(targets, particle, medium64)
	shapes = shape_mixture_per_module(targets, particle, medium64)
	
	p1 = plot(poissons[136])
	xplot = 0.01:0.1:20
	p2  = plot(xplot, loglikelihood.(shapes[136], xplot), ylim=(-10, 0))
	plot(p1, p2)
end

# ╔═╡ ec3dbcbe-028a-4700-a711-bdac55255494
begin

	function eval_loss(targets, int_grid, nph)
	
		function loss(x, y, z, log_energy)
			position = SA[x, y, z]
		    direction = SA[0.0, 0.0, 1.0]
		    time = 0.0

			particle = LightYield.Particle(
				position,
			    direction,
			    time,
			    exp10(log_energy),
			    LightYield.EMinus
			)

			precision = 0.5
			source_out = Vector{Emission.PhotonSource}(undef, size(int_grid, 1)-1)

			source_out_buf = Zygote.Buffer(source_out)
			LightYield.particle_to_elongated_lightsource!(
				particle,
				int_grid,
				medium,
				spectrum,
				(300.0, 800.0),
				source_out_buf)
			source_out = copy(source_out_buf)

			
			inputs = Matrix{Float32}(undef, (2, size(targets, 1) * size(source_out, 1)))
			inp_buf = Zygote.Buffer(inputs)
	
			for (i, (source, target)) in enumerate(product(source_out, targets))
				res = Modelling.source_to_input(source, target)
				inp_buf[1, i] = res[1]
				inp_buf[2, i] = res[2] 
			end
	
			inputs = copy(inp_buf)
			predictions = cpu(model(gpu(inputs)))
			predictions = Modelling.reverse_transformation.(predictions, output_trafos)
			predictions = reshape(predictions, (3, size(source_out, 1), size(targets, 1)))

		
			
			nph_arriv = [sum([predictions[3, i, j] * source_out[i].photons for i in 1:size(source_out, 1)]) for j in 1:size(targets, 1)]
			
			
			dists = Poisson.(nph_arriv)
			loglikelihood(dists[1], nph)
			
			
			#loglikelihood(poisson_dist_per_module(predictions, SA[source])[1], nph)
	
			#sum(loglikelihood(poisson_dist_per_module(predictions, [source])[1], nph))
			#sum(predictions[3, :, :])
		end

		Zygote.gradient(loss, 100., 1., 1., 5)
		#loss(100., 1., 1., 100)
	end

	#loss(10, 1., 80., 1.)

	len_range = (0., 30.)
	precision = 0.5
		
	int_grid = range(len_range[1], len_range[2], step=precision)
	n_steps = size(int_grid, 1)
	
	eval_loss(targets, int_grid, 20)
	
end



# ╔═╡ acfcdc3f-9582-41b4-9599-cc25ff7ecea2
uconvert(u"cm", (1u"g/cm^2" / 1u"kg/m^3"))

# ╔═╡ 06e98915-2ce6-46a3-a8bf-21640e8d202b
begin

	function diff_func(logenergy)
		particle = LightYield.Particle(
		@SVector[0., 0., 20.],
	    sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
	    0.,
	    exp10(logenergy),
	    LightYield.EMinus)

		extended_llh_per_module(ev, targets, particle, medium64)
	
	
	end

	
	
	
end

# ╔═╡ c6cfc2fd-b7eb-45a3-8dfe-3ca47db27957
begin
	ftest(x::AbstractVector{T}) where {T <: AbstractVector} = size(x)
	
	@code_warntype ftest([[1, 2, 3]])
end

# ╔═╡ 664e4d6a-93ef-40f2-a585-b3a94aa94cea
begin
	function testf(xs::T) where T
		sum(map(x -> 2*x, xs))
	end
	
	Zygote.gradient(testf, [1., 2., 3.])
end

# ╔═╡ Cell order:
# ╠═3e649cbf-1546-4204-82de-f6db5be401c7
# ╟─5ab60c75-fd19-4456-9121-fb42ce3e086f
# ╠═a615b299-3868-4083-9db3-806d7390eca1
# ╠═4db5a2bd-97bc-412c-a1d3-cb8391425e20
# ╠═b37de33c-af67-4983-9932-2c32e8db399f
# ╠═911c5c61-bcaa-4ba4-b5c0-09a112b6e877
# ╠═30c4a297-715d-4268-b91f-22d0cac66511
# ╟─26469e97-51a8-4c00-a69e-fe50ad3a625a
# ╠═74b11928-fe9c-11ec-1d37-01f9b1e48fbe
# ╠═26c7461c-5475-4aa9-b87e-7a55b82cea1f
# ╠═78695b21-e992-4f5c-8f34-28c3027e3179
# ╠═b9af02cb-b675-4030-b3c8-be866e85ebc7
# ╠═cb97110d-97b2-410d-8ce4-bef9accfcef2
# ╠═8638dde4-9f02-4dbf-9afb-32982390a0b6
# ╠═984b11c9-2cd5-4dd6-9f36-26eec69eb17d
# ╠═0a37ce94-a949-4c3d-9cd7-1a64b1a3ce47
# ╠═be1a6f87-65e3-4741-8b98-bb4d305bd8c3
# ╠═57450082-04dc-4d1a-8ef4-e321c3971c84
# ╠═31ab7848-b1d8-4380-872e-8a12c3331051
# ╠═22e419a4-a4f9-48d5-950b-8420854c475a
# ╠═4f7a9060-33dc-4bb8-9e41-eae5b9a30aa6
# ╠═40940de0-c45b-4236-82f0-54a77d5fbb9a
# ╠═709a7e6f-1e54-4350-b826-66b9097cc46a
# ╠═820e698c-f1c4-4d07-90d8-a4a13ff9cccd
# ╠═462567a7-373f-4ac6-a11a-4c6853a8c45a
# ╠═db772593-2e58-4cec-bc88-7113ac028811
# ╠═ec3dbcbe-028a-4700-a711-bdac55255494
# ╠═acfcdc3f-9582-41b4-9599-cc25ff7ecea2
# ╠═06e98915-2ce6-46a3-a8bf-21640e8d202b
# ╠═c6cfc2fd-b7eb-45a3-8dfe-3ca47db27957
# ╠═664e4d6a-93ef-40f2-a585-b3a94aa94cea
