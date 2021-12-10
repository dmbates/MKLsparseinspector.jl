module MKLsparseinspector


using Base.Enums
using LinearAlgebra
using MKL_jll
using SparseArrays

using LinearAlgebra: BlasInt

function __init__()
    ccall((:MKL_Set_Interface_Layer, libmkl_rt), Cint, (Cint,), Base.USE_BLAS64 ? 1 : 0)
end

include("enums.jl")
include("types.jl")
include("toplevel.jl")

export
    MKLcsr,
    MKLFloats,


    csrptr,
    syrkd!

end
