using Test
using Statistics
using StaticArrays
using CUDA
using Random
include("../photon_prop.jl")

using .PhotonPropagation
using .PhotonPropagationCuda
using .Utils

Random.seed!(0)
@testset "random numbers" begin
    a = 2.0
    b = 5.0
    unis = [uniform(a, b) for i in 1:10000]
    @test (minimum(unis) >= a) && (maximum(unis) <= b)

    g = 0.97
    costhetas = [cuda_hg_scattering_func(g) for i in 1:100000]
    @test isapprox(mean(costhetas), g, atol=0.01)
end

@testset "scattering" begin

    old_dir = @SVector[0.0, 0.0, 1.0]
    new_dir = @SVector[0.0, 1.0, 0.0]
    @test rotate_to_axis(old_dir, new_dir) == new_dir
    old_dir = @SVector[0.0, 0.0, -1.0]
    @test rotate_to_axis(old_dir, new_dir) == -new_dir

    function apply_rot(a, b, c)
        c[1] = rotate_to_axis(a[1], b[1])
        nothing
    end

    # Rotating to from e_z to y-axis should move y-axis to -e_z
    old_dir = @SVector[0.0, 1.0, 0.0]
    result = CuVector{SVector{3,Float64}}(undef, 1)
    @cuda apply_rot(CuVector([old_dir]), CuVector([new_dir]), result)
    @test Vector(result)[1] ≈ @SVector[0.0, 0.0, -1.0]

    a = [0.2, 0.3, 1.0]
    a .= a ./ norm(a)
    old_dir = SVector{3}(a)

    b = [-5.0, 1.0, -20.0]
    b .= b ./ norm(b)
    new_dir = SVector{3}(b)


    @cuda apply_rot(CuVector([old_dir]), CuVector([new_dir]), result)
    @test norm(Vector(result)[1]) ≈ 1.0

    cos_theta = cuda_hg_scattering_func(0.97)
    phi = uniform(0.0, 2 * pi)

    new_dir = sph_to_cart(acos(cos_theta), phi)
    @cuda apply_rot(CuVector([old_dir]), CuVector([new_dir]), result)

    rotated = Vector(result)[1]

    rotated2 = rodrigues_rotation(@SVector[0, 0, 1.0], old_dir, new_dir)
    @test rotated ≈ rotated2
end

@testset "position update" begin
    p = @SVector[0.0, 0.0, 0.0]
    a = @SVector[0.0, 1.0, 0.0]

    @test update_position(p, a, 10.0) ≈ @SVector[0, 10.0, 0]

end

@testset "interpolation" begin
    f(x) = x^2

    xs = 0.0:1.0:4.0
    ys = f.(xs)

    @test fast_linear_interp(0.5, xs, ys) ≈ 0.5
    @test fast_linear_interp(1.5, xs, ys) ≈ 2.5
    @test fast_linear_interp(4.5, xs, ys) ≈ 16
    @test fast_linear_interp(-1., xs, ys) ≈ 0

    @test fast_linear_interp(0.5, ys, 0.0, 4.0) ≈ 0.5
    @test fast_linear_interp(1.5, ys, 0.0, 4.0) ≈ 2.5
    @test fast_linear_interp(4.5, ys, 0.0, 4.0) ≈ 16
    @test fast_linear_interp(-1., ys, 0.0, 4.0) ≈ 0
end