using MKLsparseinspector
using LinearAlgebra
using SparseArrays
using Test

using LinearAlgebra: BlasInt

@testset "MKLcsr" begin
    spm = sparse(BlasInt.(collect(1:6)), BlasInt[3,2,3,1,3,3], ones(6))
    csrpt = csrptr(spm')
    @test isa(csrpt, Ptr{MKLcsr{Float64}})
    @test SparseMatrixCSC(csrpt) == spm  # this may change
end
