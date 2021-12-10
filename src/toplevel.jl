function syrkd!(C::StridedMatrix{T}, A::SparseMatrixCSC{T,BlasInt}, op::Char, α::Number, β::Number) where {T}
    return syrkd!(C, csrptr(A'), op, α, β)
end
