using StaticArrays
using BenchmarkTools
using LinearAlgebra
using CUDA
using StatsPlots
using BenchmarkPlots
using Adapt
using Random
using Cthulhu


@inline function uniform(minval::T, maxval::T) where {T}
    uni = rand(T)
    return minval + uni * (maxval - minval)
end

@inline function cuda_scattering_func(g::T) where {T}
    """Henyey-Greenstein scattering in one plane."""
    eta = rand(T)
    costheta::T = (1 / (2 * g) * (1 + g^2 - ((1 - g^2) / (1 + g * (2 * eta - 1)))^2))
    CUDA.acos(CUDA.clamp(costheta, T(-1), T(1)))
end





#iso_mono_source = (initialize_direction_isotropic, (::Type) -> @SVector[0f0, 0f0, 0f0], (::Type) -> 0f0, (::Type) -> 450f0)




@inline function _update_direction!(this_dir::AbstractArray{T}) where {T}

    # Calculate new direction (relative to e_z)
    sca_theta = cuda_scattering_func(T(0.99))
    sca_phi = uniform(T(0), T(2 * pi))

    cos_theta = CUDA.cos(sca_theta)
    sin_theta = CUDA.sin(sca_theta)

    sin_phi = CUDA.sin(sca_phi)
    cos_phi = CUDA.cos(sca_phi)


    if CUDA.abs(this_dir[3]) == 1.0f0
        sign = CUDA.sign(this_dir[3])

        this_dir[1] = newsin_theta * cos_phix
        this_dir[2] = sign * sin_theta * sin_phi
        this_dir[3] = sign * cos_theta
        return
    end

    denom = CUDA.sqrt(1.0 - this_dir[3]^2)
    muzcosphi = this_dir[3] * cos_phi
    new_x = sin_theta * (this_dir[1] * muzcosphi - this_dir[2] * sin_phi) / denom + this_dir[1] * cos_theta
    new_y = sin_theta * (this_dir[2] * muzcosphi + this_dir[1] * sin_phi) / denom + this_dir[2] * cos_theta
    new_z = -denom * sin_theta * cos_phi + this_dir[3] * cos_theta

    norm = CUDA.sqrt(new_x^2 + new_y^2 + new_z^2)
    this_dir[1] = new_x / norm
    this_dir[2] = new_y / norm
    this_dir[3] = new_z / norm
    return

end


@inline function update_direction!(this_dir::AbstractArray{T}) where {T}
    """
    Update the photon direction using scattering function.

    New direction is relative to e_z. Axis of rotation defined by rotating e_z to old dir and applying
    that transformation to new_dir.

    Rodrigues rotation formula:
        ax = e_z x dir
        #theta = acos(dir * new_dir)
        theta = asin(dir x new_dir)
        
        axop = axis x new_dir
        rotated = new_dir * cos(theta) + sin(theta) * (axis x new_dir) + (1-cos(theta)) * (axis * new_dir) * axis

    """


    # Calculate new direction (relative to e_z)
    sca_theta = cuda_scattering_func(T(0.99))
    sca_phi = uniform(T(0), T(2 * pi))

    new_dir_1 = CUDA.cos(sca_phi) * CUDA.sin(sca_theta)
    new_dir_2 = CUDA.sin(sca_phi) * CUDA.sin(sca_theta)
    new_dir_3 = CUDA.cos(sca_theta)


    if CUDA.abs(this_dir[3]) == 1.0f0

        sign = CUDA.sign(this_dir[3])
        this_dir[1] = new_dir_1
        this_dir[2] = sign * new_dir_2
        this_dir[3] = sign * new_dir_3

        return
    end

    # Determine axis of rotation (cross product of e_z and old_dir )    
    ax1 = -this_dir[2]
    ax2 = this_dir[1]

    # Determine angle of rotation (cross product e_z and old_dir)
    # sin(theta) = | e_z x old_dir |

    sinthetasq = 1 - this_dir[3]^2
    costheta = CUDA.sqrt(1 - sinthetasq)
    sintheta = CUDA.sqrt(sinthetasq)


    # cross product of axis with new_ direction
    axop1 = ax2 * new_dir_3
    axop2 = -ax1 * new_dir_3
    axop3 = ax1 * new_dir_2 - ax2 * new_dir_1

    axopdot = ax1 * new_dir_1 + ax2 * new_dir_2

    new_x = new_dir_1 * costheta + axop1 * sintheta + ax1 * axopdot * (1 - costheta)
    new_y = new_dir_2 * costheta + axop2 * sintheta + ax2 * axopdot * (1 - costheta)
    new_z = new_dir_3 * costheta + axop3 * sintheta

    norm = CUDA.sqrt(new_x^2 + new_y^2 + new_z^2)

    this_dir[1] = new_x / norm
    this_dir[2] = new_y / norm
    this_dir[3] = new_z / norm
    return

end


@inline function update_position(this_pos, this_dir, this_dist_travelled, step_size)

    # update position
    for j in Int32(1):Int32(3)
        this_pos[j] = this_pos[j] + this_dir[j] * step_size
    end

    this_dist_travelled[1] += step_size
    return nothing
end


function cuda_step_photons!(
    positions::CuDeviceMatrix{T},
    directions::CuDeviceMatrix{T},
    dist_travelled::CuDeviceArray{T},
    sca_coeffs::CuDeviceArray{T},
    intersected::CuDeviceArray{Bool},
    ::Val{Target},
    ::Val{Steps},
    seed::UInt32) where {T,Target,Steps}

    target::PhotonTarget{T} = Target
    steps::UInt16 = Steps

    block = UInt32(blockIdx().x)
    thread = UInt32(threadIdx().x)
    blockdim = UInt32(blockDim().x)
    griddim = UInt32(gridDim().x)

    arraysize = UInt32(size(positions, 2))

    index::UInt32 = (block - Int32(1)) * blockdim + thread
    stride::UInt32 = griddim * blockdim
    Random.seed!(seed + index)


    cache_len::Int32 = 7
    cache = @cuDynamicSharedMem(T, cache_len * blockdim)


    @inbounds for i = index:stride:arraysize

        cache_ix_offset = cache_len * (thread - Int32(1))
        #this_photon = view(cache, 
        this_pos = view(cache, cache_ix_offset+Int32(1):cache_ix_offset+Int32(3))
        this_dir = view(cache, cache_ix_offset+Int32(4):cache_ix_offset+Int32(6))
        this_dist_travelled = view(cache, cache_ix_offset+7:cache_ix_offset+7)

        for j in 1:3
            this_pos[j] = positions[j, i]
            this_dir[j] = directions[j, i]
        end
        this_dist_travelled[1] = dist_travelled[i]
        sca_coeff = sca_coeffs[i]
        this_intersected = view(intersected, i:i)


        for nstep in UInt16(1):steps

            eta = rand(T)
            step_size::Float32 = -CUDA.log(eta) / sca_coeff

            # Check intersection with module

            # a = dot(dir, (pos - target.position))
            # pp_norm_sq = norm(pos - target_pos)^2
            a::Float32 = 0.0f0
            pp_norm_sq::Float32 = 0.0f0
            for j in Int32(1):Int32(3)
                a += this_dir[j] * (this_pos[j] - target.position[j])
                pp_norm_sq += (this_pos[j] - target.position[j])^2
            end

            b::Float32 = a^2 - (pp_norm_sq - target.radius^2)

            isec = b >= 0

            if isec
                # Uncommon branch
                # Distance of of the intersection point along the line
                d = -a - CUDA.sqrt(b)

                if (d > 0) & (d < step_size)
                    # Step to intersection
                    this_intersected[1] = true

                    #set new position
                    update_position(this_pos, this_dir, this_dist_travelled, d)
                    continue
                end
            end
            update_position(this_pos, this_dir, this_dist_travelled, step_size)
            update_direction!(this_dir)

        end


        for j in 1:3
            positions[j, i] = this_pos[j]
            directions[j, i] = this_dir[j]
        end
        dist_travelled[i] = this_dist_travelled[1]
    end


    nothing
end



function sph_to_cart(theta::T, phi::T) where {T}
    x::T = cos(phi) * sin(theta)
    y::T = sin(phi) * sin(theta)
    z::T = cos(theta)

    [x, y, z]
end





function calc_shmem(block_size)
    block_size * 7 * sizeof(Float32) #+ 3 * sizeof(Float32)
end



function propagate(photons, intersected, photon_target, steps, seed)
    kernel = @cuda launch = false cuda_step_photons!(photons, intersected, Val(photon_target), Val(steps), seed)
    config = launch_configuration(kernel.fun, shmem=calc_shmem)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    all_positions = Array{Float32}(undef, 3, N, steps + 1)
    all_positions[:, :, 1] = Array(photons[1:3, :])

    for i in 1:steps
        kernel(photons, intersected, Val(photon_target), Val(1), seed + i * N, threads=threads, blocks=blocks, shmem=calc_shmem(threads))
        all_positions[:, :, i+1] = Array(photons[1:3, :])
    end
    all_positions
end



function make_bench_cuda_step_photons!(N)
    sca_len = 10.0f0
    target_pos = @SVector [0.0f0, 0.0f0, 5.0f0]
    target = PhotonTarget(target_pos, 1.0f0)

    photons = initialize_photons(N, Float32, (T) -> [0.0f0, 0.0f0, 0.0f0], initialize_direction_isotropic, (T) -> 1 / 20.0f0)
    intersected = CuArray(zeros(Bool, N))

    steps = UInt16(10)
    seed = UInt32(1)

    pos = CuArray(photons[1:3, :])
    dir = CuArray(photons[4:6, :])
    dist_travelled = CuArray(photons[7, :])
    sca_coeffs = CuArray(photons[8, :])

    kernel = @cuda launch = false cuda_step_photons!(
        pos,
        dir,
        dist_travelled,
        sca_coeffs,
        intersected,
        Val(target),
        Val(steps),
        seed)
    config = launch_configuration(kernel.fun, shmem=calc_shmem)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    println("N: $N, threads: $threads, blocks: $blocks")
    shmem = calc_shmem(threads)
    bench = @benchmarkable CUDA.@sync $kernel(
        $pos, $dir, $dist_travelled, $sca_coeffs, $intersected, $(Val(target)), $(Val(steps)), $seed, threads=$threads, blocks=$blocks, shmem=$shmem)
    CUDA.reclaim()
    bench
end


function test()
    N = 100
    sca_len = 10.0f0
    target_pos = @SVector [0.0f0, 0.0f0, 5.0f0]
    target = PhotonTarget(target_pos, 1.0f0)
    pos, dir, intersected, dist_travelled = generate_inputs(N)
    steps = UInt16(10)
    seed = UInt32(1)
    # @device_code_llvm
    photon_config = PhotonConfig(target, 1 / sca_len, 1.0f0)
    kernel = @cuda launch = false cuda_step_photons!(pos, dir, intersected, dist_travelled, Val(photon_config), Val(steps), seed)
    config = launch_configuration(kernel.fun, shmem=calc_shmem)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    println("N: $N, threads: $threads, blocks: $blocks")
    kernel(pos, dir, intersected, dist_travelled, photon_config, steps, seed, threads=threads, blocks=blocks, shmem=calc_shmem(threads))
    # kernel(pos, dir, intersected, target, sca_len, steps, state, threads, blocks, shmem=calc_shmem(threads))
    #end
end

#test()


#CUDA.@device_code_warntype interactive = true test()
#test()

"""
suite = BenchmarkGroup()

nphs = trunc.(Int, 10 .^ (range(1, 7.5, length=10)))

for nph in nphs
    suite[nph] = make_bench_cuda_step_photons!(nph)
end

results = run(suite, verbose=true)
med_time_per_ph = Dict(key => val.time / key for (key, val) in median(results))
print(med_time_per_ph)
plot(med_time_per_ph, yscale=:log10, ylabel="Time / photon", xlabel="Photons", ylims=(0.1, 1E3), xscale=:log10)

#test()
"""
