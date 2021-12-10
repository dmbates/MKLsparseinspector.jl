MKLFloats = Union{Float32,Float64,ComplexF32,ComplexF64}

"""
    MKLcsr{T<:MKLFloats}

The opaque struct `sparse_matrix` from the MKL library in compressed sparse row (CSR) format
"""
mutable struct MKLcsr{T<:MKLFloats}
end

Base.eltype(m::MKLcsr{T}) where {T} = T
 

for (T, cr, ex, syrk) in (
    (Float32,    :mkl_sparse_s_create_csr, :mkl_sparse_s_export_csr, :mkl_sparse_s_syrkd,),
    (Float64,    :mkl_sparse_d_create_csr, :mkl_sparse_d_export_csr, :mkl_sparse_d_syrkd,),
    (ComplexF32, :mkl_sparse_c_create_csr, :mkl_sparse_c_export_csr, :mkl_sparse_c_syrkd,),
    (ComplexF64, :mkl_sparse_z_create_csr, :mkl_sparse_z_export_csr, :mkl_sparse_z_syrkd,),
    )
    @eval begin
        function csrptr(adjm::Adjoint{$T,SparseMatrixCSC{$T,BlasInt}})
            m = adjm.parent
            r = Ref(Ptr{MKLcsr{$T}}(0))
            ret = ccall(
                ($(string(cr)), libmkl_rt),
                sparse_status_t,
                (
                    Ref{Ptr{MKLcsr{$T}}}, sparse_index_base_t, BlasInt, BlasInt, Ptr{BlasInt},
                    Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}
                ),
                r, SPARSE_INDEX_BASE_ONE, m.n, m.m, m.colptr, pointer(m.colptr, 2),
                m.rowval, m.nzval
            )
            ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
            p = r[]
            value = unsafe_load(p)
            finalizer(value) do x
                ccall(
                    (:mkl_sparse_destroy, libmkl_rt),
                    sparse_status_t,
                    (Ref{MKLcsr{$T}},),
                    Ref(value),
                )
            end
            return p
        end

        function SparseArrays.SparseMatrixCSC(csrpt::Ptr{MKLcsr{$T}})
            indexing = Ref(Cint(0))
            rows = Ref(BlasInt(0))
            cols = Ref(BlasInt(0))
            rows_start = Ref(Ptr{BlasInt}(0))
            rows_end = Ref(Ptr{BlasInt}(0))
            col_indx = Ref(Ptr{BlasInt}(0))
            values = Ref(Ptr{$T}(0))
            ret = ccall(
                ($(string(ex)), libmkl_rt),
                sparse_status_t,
                (Ptr{MKLcsr{$T}}, Ref{Cint}, Ref{BlasInt}, Ref{BlasInt},
                Ref{Ptr{BlasInt}}, Ref{Ptr{BlasInt}}, Ref{Ptr{BlasInt}}, Ref{Ptr{$T}}),
                csrpt, indexing, rows, cols, rows_start, rows_end, col_indx, values,
            )
            ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(ret))
            rowptr = Vector{BlasInt}(undef, rows[] + 1)
            unsafe_copyto!(pointer(rowptr), rows_start[], rows[])
            rowend = Vector{BlasInt}(undef, rows[])
            unsafe_copyto!(pointer(rowend), rows_end[], rows[])
            nonzeros = last(rowend) - indexing[]
            first(rowptr) == indexing[] || throw(ArgumentError("indexing = $indexing ≠ $(first(rowptr)) = first(rowptr)"))
            view(rowptr, 2:rows[]) == view(rowend, 1:(rows[]-1)) || throw(ArgumentError("indices are not dense"))
            rowptr[rows[] + 1] = last(rowend)
            colvals = Vector{BlasInt}(undef, nonzeros)
            unsafe_copyto!(pointer(colvals), col_indx[], nonzeros)
            nzvals = Vector{$T}(undef, nonzeros)
            unsafe_copyto!(pointer(nzvals), values[], nonzeros)
            return SparseMatrixCSC{$T,BlasInt}(cols[], rows[], rowptr, colvals, nzvals)     
        end

        function syrkd!(C::StridedMatrix{$T}, Apt::Ptr{MKLcsr{$T}}, op::Char, α::Number, β::Number)
            ret = ccall(
                ($(string(syrk)), libmkl_rt),
                sparse_status_t,
                (sparse_operation_t, Ptr{MKLcsr{$T}}, $T, $T, Ptr{$T}, sparse_layout_t, BlasInt),
                uppercase(op) == 'T' ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
                Apt, α, β, C, SPARSE_LAYOUT_COLUMN_MAJOR, stride(C, 2),
            )
            ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(ret))
            return C
        end
    end # eval        
end #loop on types
