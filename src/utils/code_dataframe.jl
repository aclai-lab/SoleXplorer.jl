function code_dataframe(X::AbstractDataFrame, y::AbstractVector)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = @. CategoricalArrays.levelcode(col) 
            X[!, name] = X_coded
        end
    end

    if !(eltype(y) <: Number)
        y = @. CategoricalArrays.levelcode(y) 
    end
    
    return X, y
end

# TODO separa X da y