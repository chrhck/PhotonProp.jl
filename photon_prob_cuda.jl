using StaticArrays
using BenchmarkTools
using LinearAlgebra
using CUDA
using StatsPlots
using BenchmarkPlots
using Adapt
#using GPUArrays


mutable struct RNGState
    state::UInt32
end

@inline function lcg_parkmiller!(state::RNGState)

    A::UInt32 = 48271

    low::UInt32 = (state.state & 0x7fff) * A# max: 32,767 * 48,271 = 1,581,695,857 = 0x5e46c371
    high::UInt32 = (state.state >> 15) * A# max: 65,535 * 48,271 = 3,163,439,985 = 0xbc8e4371
    x::UInt32 = low + ((high & 0xffff) << 15) + (high >> 16)# max: 0x5e46c371 + 0x7fff8000 + 0xbc8e = 0xde46ffff

    x = (x & 0x7fffffff) + (x >> 31)
    state.state = x
end

cuda_update_rng_state!(state::RNGState) = lcg_parkmiller!(state)

function cuda_uniform!(state::RNGState)
    val = state.state
    uniform::Float32 = Float32(val) / typemax(Int32)
    cuda_update_rng_state!(state)

    return uniform
end

function cuda_uniform!(state::RNGState, minval::Float32, maxval::Float32)
    minval + cuda_uniform!(state) * (maxval - minval)
end




struct PhotonTarget{T<:Real}
    position::SVector{3,T}
    radius::T
end

mutable struct RNGState
    val::UInt32
end


abstract type HolderType{T} end

struct Steps{T} <: HolderType{T}
    Steps(x::T) where {T<:Integer} = new{x}()
end
Steps(x) = throw(DomainError("noninteger type"))

struct TargetHolder{T} <: HolderType{T}
    #PhotonTarget(x::Position, y::Radius) where {T, Position <: SVector{3, T}, Radius <: T} = new{Position, Radius}()
    TargetHolder(x::T) where {U,T<:PhotonTarget{U}} = new{x}()
end

access_type_var(th::HolderType{T}) where {T} = T


function cuda_step_photons!(
    positions::CuDeviceMatrix{T},
    directions::CuDeviceMatrix{T},
    intersected::CuDeviceArray{Bool},
    target_holder::TargetHolder{PhTT},
    #target::Target
    sca_coeff::T,
    steps_holder::Steps{steps},
    seed::UInt32) where {T,steps,PhTT}

    target::PhotonTarget{T} = access_type_var(target_holder)

    block = Int32(blockIdx().x)
    thread = Int32(threadIdx().x)
    blockdim = Int32(blockDim().x)
    griddim = Int32(gridDim().x)

    arraysize = Int32(size(directions, 2))

    index = (block - Int32(1)) * blockdim + thread
    stride = griddim * blockdim
    state = RNGState(seed + index)

    cache_len::Int32 = 6

    cache = @cuDynamicSharedMem(T, cache_len * blockdim)

    @inbounds for i = index:stride:arraysize

        cache_ix_offset = cache_len * (thread - Int32(1))

        this_pos = view(cache, cache_ix_offset+Int32(1):cache_ix_offset+Int32(3))
        this_dir = view(cache, cache_ix_offset+Int32(4):cache_ix_offset+Int32(6))


        for j in Int32(1):Int32(3)
            this_pos[j] = positions[j, i]
            this_dir[j] = directions[j, i]
        end

        for nstep in UInt16(1):steps

            eta = cuda_uniform!(state)
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

            if b < 0
                isec = false
            else
                # Distance of of the intersection point along the line
                d = -a - CUDA.sqrt(b)
                isec = (b >= 0) & (d > 0) & (d < step_size)
            end

            if isec
                # Step to intersection
                step_size = d
            end

            #set new position

            for j in Int32(1):Int32(3)
                this_pos[j] = this_pos[j] + this_dir[j] * step_size
            end

            if isec
                intersected[i] = true
                break
            end

            sca_theta = cuda_scattering_func(T(0.99), state)
            sca_phi = cuda_uniform!(state) * T(2 * pi)

            new_dir_1 = CUDA.cos(sca_phi) * CUDA.sin(sca_theta)
            new_dir_2 = CUDA.sin(sca_phi) * CUDA.sin(sca_theta)
            new_dir_3 = CUDA.cos(sca_theta)


            # Rodrigues rotation formula
            # axis = e_z x dir
            # theta = acos(dir * new_dir)
            # axop = axis x new_dir
            # rotated = new_dir * cos(theta) + sin(theta) * (axis x new_dir) + (1-cos(theta)) * (axis * new_dir) * axis
            # New direction is relative to e_z. Axis of rotation defined by rotating e_z to old dir and applying
            # that transformation to new_dir.

            # cross product with e_z
            ax1 = this_dir[2]
            ax2 = this_dir[1]
            #ax3 = 0

            # cross product of axis with operand (e)
            axop1 = ax2 * new_dir_3
            axop2 = -ax1 * new_dir_3
            axop3 = ax1 * new_dir_2 - ax2 * new_dir_1

            costheta = (
                this_dir[1] * new_dir_1 +
                this_dir[2] * new_dir_2 +
                this_dir[3] * new_dir_3
            )
            sintheta = CUDA.sqrt(1 - costheta^2)

            axopdot = ax1 * new_dir_1 + ax2 * new_dir_2

            this_dir[1] = new_dir_1 * costheta + axop1 * sintheta + ax1 * axopdot * (1 - costheta)
            this_dir[2] = new_dir_2 * costheta + axop2 * sintheta + ax2 * axopdot * (1 - costheta)
            this_dir[3] = new_dir_3 * costheta + axop3 * sintheta
        end

        # Load back from cache
        for j in Int32(1):Int32(3)
            positions[j, i] = this_pos[j]
            directions[j, i] = this_dir[j]
        end

    end
    nothing
end




function norm2(A; dims)
    B = sum(x -> x^2, A; dims=dims)
    B .= sqrt.(B)
end


function generate_inputs(N::Integer)
    positions = zeros(Float32, 3, N)
    #fill!(positions, SVector(0.0f0, 0.0f0, 0.0f0))
    dirs = Matrix{Float32}(undef, 3, N)
    for i in 1:N
        dirs[:, i] = [0, 0, 1.0f0]
    end

    intersected = zeros(Bool, N)

    #fill!(dirs, SVector(0.0f0, 0.0f0, 1.0f0))
    (CuArray(positions), CuArray(dirs), CuArray(intersected))
end


function calc_shmem(block_size)
    block_size * 6 * sizeof(Float32) + 3 * sizeof(Float32)
end

function propagate(pos, dir, intersected, target, sca_coeff, steps, seed)
    kernel = @cuda launch = false cuda_step_photons!(pos, dir, intersected, target, sca_coeff, steps, seed)
    config = launch_configuration(kernel.fun, shmem=calc_shmem)
    threads = min(N, config.threads)
    blocks = cld(N, threads)


    all_positions = Array{Float32}(undef, 3, N, steps)
    all_positions[:, :, 1] = Array(pos)

    for i in 1:steps
        kernel(pos, dir, intersected, target, sca_coeff, 1, seed + i * N, threads=threads, blocks=blocks, shmem=calc_shmem(threads))
        println(pos)
        println(dir)
        println(intersected)
        all_positions[:, :, i] = Array(pos)
    end

    all_positions
end



function make_bench_cuda_step_photons!(N)
    sca_len = 10.0f0
    target_pos = @SVector [0.0f0, 0.0f0, 5.0f0]
    target = PhotonTarget(target_pos, 1.0f0)
    pos, dir, intersected = generate_inputs(N)
    steps = Steps{1}()
    seed = UInt32(1)
    kernel = @cuda launch = false cuda_step_photons!(pos, dir, intersected, target, 1 / sca_len, steps, seed)
    config = launch_configuration(kernel.fun, shmem=calc_shmem)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    println("N: $N, threads: $threads, blocks: $blocks")
    shmem = calc_shmem(threads)
    bench = @benchmarkable CUDA.@sync $kernel(
        $pos, $dir, $intersected, $target, $sca_len, $steps, $seed, threads=$threads, blocks=$blocks, shmem=$shmem)
    CUDA.reclaim()
    bench
end


function test()
    sca_len = 10.0f0
    target_pos = @SVector [0.0f0, 0.0f0, 5.0f0]
    target = PhotonTarget(target_pos, 1.0f0)
    N = 1000000
    pos, dir, intersected = generate_inputs(N)
    steps::UInt16 = 30
    seed = UInt32(1)
    kernel = @cuda launch = false cuda_step_photons!(pos, dir, intersected, target, 1 / sca_len, steps, seed)
    config = launch_configuration(kernel.fun, shmem=calc_shmem)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    print("Threads: $threads, Blocks: $blocks")
    #CUDA.@profile begin
    kernel(pos, dir, intersected, target, sca_len, steps, seed, threads=threads, blocks=blocks, shmem=calc_shmem(threads))
    # kernel(pos, dir, intersected, target, sca_len, steps, state, threads, blocks, shmem=calc_shmem(threads))
    #end
end



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
