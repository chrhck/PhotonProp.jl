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
end

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
@df prop_result corrplot(cols(1:4))

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

Note: The detector is assumed to be 100% efficient. Lower detection efficiences can be trivialled added by aan additional (wl-dependent) weight.

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
	#results_df = Modelling.make_photon_fits(Int64(1E8), 150, 150, 300f0)
	#write_parquet("photon_fits.parquet", results_df)
	results_df = read_parquet("photon_fits.parquet")
end


# ╔═╡ 26c7461c-5475-4aa9-b87e-7a55b82cea1f
@df results_df corrplot(cols(1:5))

# ╔═╡ b9af02cb-b675-4030-b3c8-be866e85ebc7
md"""
## Fit distribution parameters with MLP
Here we fit a simple MLP to predict the distribution parameters (and the photon survival rate) as function of the distance and the observation angle
"""

# ╔═╡ cb97110d-97b2-410d-8ce4-bef9accfcef2
begin
	model, test_data = Modelling.train_mlp(epochs=400, width=600, learning_rate=0.001, batch_size=300, data_file="photon_fits.parquet",
	dropout_rate=0.5)
	@show Modelling.loss_all(test_data, model)
end

# ╔═╡ 8638dde4-9f02-4dbf-9afb-32982390a0b6
begin
	feat_test = hcat(collect(td[1] for td in test_data)...)
	targ_test = hcat(collect(td[2] for td in test_data)...)
	target_pred = model(feat_test)
	feat_test = cpu(feat_test)
	targ_test = cpu(targ_test)
	target_pred = cpu(target_pred)
	
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
	
end

# ╔═╡ 0a37ce94-a949-4c3d-9cd7-1a64b1a3ce47
md"""
## Compare model prediction to MC
"""

# ╔═╡ be1a6f87-65e3-4741-8b98-bb4d305bd8c3
begin
    position = @SVector[0.0, 0.0, 0.0]
    direction = @SVector[0.0, 0.0, 1.0]
    energy = 1E3
    time = 0.0
	photons = 1000000
	medium64 = Medium.make_cascadia_medium_properties(Float64)
	spectrum = Spectral.CherenkovSpectrum((300.0u"nm", 800.0u"nm"), 20, medium64)
	em_profile = Emission.AngularEmissionProfile{:IsotropicEmission, Float64}()
	
	source = Emission.PhotonSource(position, direction, time, photons, spectrum, em_profile)

	target_pos = @SVector[0.00, 80.0, 10.0]
	target = Detection.DetectionSphere(target_pos, 0.21)
	
	model_input = Matrix{Float32}(undef, (2, 1))
	model_input[:, 1] .= Modelling.source_to_input(source, target)
	model_pred = cpu(model(gpu(model_input)))
	n_photons_pred = photons * 10^(-model_pred[3])

	em_rec_vec = source.position .- target.position
	obs_angle = acos(dot(em_rec_vec ./ norm(em_rec_vec), source.direction))

	this_prop_result = PhotonPropagation.propagate_distance(Float32(norm(em_rec_vec)), medium, photons)

	this_total_weight = Modelling.get_dir_reweight(
		this_prop_result[:, :initial_theta],
		obs_angle,
		this_prop_result[:, :ref_ix]) .* this_prop_result[:, :abs_weight]
	
	histogram(this_prop_result[:, :tres], weights=this_total_weight, normalize=false)

	xs_plot = 0:0.1:15
	
	gamma_pdf = Gamma(model_pred[1], model_pred[2])
	
	plot!(xs_plot, n_photons_pred .* pdf.(gamma_pdf, xs_plot))
	n_photons_pred, sum(this_total_weight)
end





# ╔═╡ 31ab7848-b1d8-4380-872e-8a12c3331051
md"""
## Use model to simulate an event for a detector
"""

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

# ╔═╡ 910f5ee0-9859-4861-abc7-787648ae4c97


# ╔═╡ 0dde21b1-6f84-4b1e-a8fa-82741ff90e13
begin
	targets = make_targets(positions)
	particle = LightYield.Particle(
		@SVector[0., 0., 0.],
	    @SVector[0., 0., 1.],
	    0.,
	    1E5,
	    LightYield.EMinus
	)
	
	sources = LightYield.particle_to_elongated_lightsource(
		particle,
		(0.0u"m", 30.0u"m"),
		0.1u"m",
		medium64,
		(250.0u"nm", 800.0u"nm"))
	inputs = Modelling.source_to_input(sources, targets)
	predictions = cpu(model(gpu(inputs)))

end






# ╔═╡ 0c0f4d2f-188f-433b-84aa-8258d44e3bdf
begin
	distances = norm.([src.position .- tgt.position for (src, tgt) in product(sources, targets)])[:]
	
	scatter(distances, 10 .^(-predictions[3, :]), xscale=:log10, yscale=:log10)
end

# ╔═╡ Cell order:
# ╠═3e649cbf-1546-4204-82de-f6db5be401c7
# ╟─5ab60c75-fd19-4456-9121-fb42ce3e086f
# ╠═a615b299-3868-4083-9db3-806d7390eca1
# ╠═4db5a2bd-97bc-412c-a1d3-cb8391425e20
# ╠═b37de33c-af67-4983-9932-2c32e8db399f
# ╟─911c5c61-bcaa-4ba4-b5c0-09a112b6e877
# ╠═30c4a297-715d-4268-b91f-22d0cac66511
# ╟─26469e97-51a8-4c00-a69e-fe50ad3a625a
# ╠═74b11928-fe9c-11ec-1d37-01f9b1e48fbe
# ╠═26c7461c-5475-4aa9-b87e-7a55b82cea1f
# ╠═b9af02cb-b675-4030-b3c8-be866e85ebc7
# ╠═cb97110d-97b2-410d-8ce4-bef9accfcef2
# ╠═8638dde4-9f02-4dbf-9afb-32982390a0b6
# ╠═0a37ce94-a949-4c3d-9cd7-1a64b1a3ce47
# ╠═be1a6f87-65e3-4741-8b98-bb4d305bd8c3
# ╠═31ab7848-b1d8-4380-872e-8a12c3331051
# ╠═4f7a9060-33dc-4bb8-9e41-eae5b9a30aa6
# ╠═40940de0-c45b-4236-82f0-54a77d5fbb9a
# ╠═910f5ee0-9859-4861-abc7-787648ae4c97
# ╠═0dde21b1-6f84-4b1e-a8fa-82741ff90e13
# ╠═0c0f4d2f-188f-433b-84aa-8258d44e3bdf
