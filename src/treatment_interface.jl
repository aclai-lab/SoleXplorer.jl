function get_treatment(X::DataFrame, model::T, features::AbstractVector; 
    vnames::Union{AbstractVector{Union{String, Symbol}}, Nothing}=nothing,
    nwindows::Union{Int, Nothing}=nothing,
    relative_overlap::Union{Float64, Nothing}=nothing
) where {T<:SoleXplorer.ModelConfig}
    # ------------------------------------------------------------------------ #
    #                                data check                                #
    # ------------------------------------------------------------------------ #
    isnothing(nwindows)         && model.data_treatment == :reducesize       && (nwindows = 10)
    isnothing(nwindows)         && model.data_treatment == :reduce_aggregate && (nwindows = 4)
    isnothing(relative_overlap) && model.data_treatment == :reducesize       && (relative_overlap = 0.2)
    isnothing(relative_overlap) && model.data_treatment == :reduce_aggregate && (relative_overlap = 0.0)

    if !isnothing(vnames)
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames = eltype(vnames) <: Symbol ? string.(vnames) : vnames
    else
        vnames = names(X)
    end

    # ------------------------------------------------------------------------ #
    #                                treatment                                 #
    # ------------------------------------------------------------------------ #
    if model.data_treatment == :aggregate
        valid_X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")") for j in features for i in vnames]])
        push!(valid_X, [vcat([map(f, Array(row)) for f in features]...) for row in eachrow(X)]...)

    elseif model.data_treatment == :reducesize
        valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
        # TODO sparito da SoleBase, dobe sarà?
        # push!(valid_X, collect(SoleBase.movingaverage.(Array(X[i, :]); nwindows = nwindows, relative_overlap = relative_overlap) for i in 1:nrow(X))...)
        push!(valid_X, collect(movingaverage.(Array(X[i, :]); nwindows = nwindows, relative_overlap = relative_overlap) for i in 1:nrow(X))...)
        
    elseif model.data_treatment == :reduce_aggregate
        valid_X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")w", k) for j in features for i in vnames for k in 1:nwindows]])
        # TODO sparito da SoleBase, dobe sarà?
        # win_X = [vcat(SoleBase.movingwindow.(Array(row); nwindows = nwindows, relative_overlap = relative_overlap)...) for row in eachrow(X)]
        win_X = [vcat(movingwindow.(Array(row); nwindows = nwindows, relative_overlap = relative_overlap)...) for row in eachrow(X)]
        push!(valid_X, [vcat([map(f, i) for f in features for i in row]...) for row in eachrow(win_X)]...)
    end

    return valid_X
end