module Detection
using StaticArrays

export PhotonTarget, DetectionSphere

abstract type PhotonTarget{T<:Real} end

struct DetectionSphere{T<:Real} <: PhotonTarget{T}
    position::SVector{3,T}
    radius::T
end
end