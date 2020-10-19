const MKLFloats = Union{Float32,Float64,ComplexF32,ComplexF64}

struct mklsparsecsr{T} where {T<:MKLFloats}
    r::Ref{Ptr{Cvoid}}
    desc::matrix_descr
end

const createcsr = Dict(
    Float32    => :mkl_sparse_s_create_csr,
    Float64    => :mkl_sparse_d_create_csr,
    ComplexF32 => :mkl_sparse_c_create_csr,
    ComplexF64 => :mkl_sparse_z_create_csr,
)

function mklsparsecsr(a::SparseMatrixCSC{T,BlasInt}) where {T<:MKLFloats}
    p = Ref(Ptr{Cvoid}(0))
    ret = ccall(
        (createcsr[T], libmkl_rt),
        sparse_status_t,
        (Ref{Ptr{Cvoid}}, sparse_index_base_t, BlasInt, BlasInt, Ptr{BlasInt},
        Ptr{BlasInt}, Ptr{BlasInt}, Ptr{T}),
        p, SPARSE_INDEX_BASE_ONE, m.m, m.n, m.colptr, pointer(m.colptr, 2), m.rowval,
        m.nzval)
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
    mklsparsecsr{T}(
        p, 
        matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT),
    )
end
