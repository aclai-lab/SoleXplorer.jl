# ---------------------------------------------------------------------------- #
#                               get treatment                                  #
# ---------------------------------------------------------------------------- #

function get_treatment(X::DataFrame, model::T, features::AbstractVector; 
    vnames::Union{AbstractVector{Union{String, Symbol}}, Nothing}=nothing,
    treatment::Union{Symbol, Nothing}=nothing,
    nwindows::Union{Int, AbstractFloat, Nothing}=nothing,
    overlap::Union{Int, AbstractFloat, Nothing}=nothing,
) where {T<:SoleXplorer.ModelConfig}
    # ------------------------------------------------------------------------ #
    #                       check sub features names                           #
    # ------------------------------------------------------------------------ #
    if !isnothing(vnames)
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames = eltype(vnames) <: Symbol ? string.(vnames) : vnames
    else
        vnames = names(X)
    end

    if isnothing(treatment)
        # -------------------------------------------------------------------- #
        #                               treatment                              #
        # -------------------------------------------------------------------- #
        if model.data_treatment == :aggregate
            valid_X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")") for j in features for i in vnames]])
            push!(valid_X, [vcat([map(f, Array(row)) for f in features]...) for row in eachrow(X)]...)

        elseif model.data_treatment == :reducesize
            isnothing(nwindows) && (nwindows = 10)
            isnothing(overlap) && (overlap = 0.2)
            overlap isa Int && (overlap = overlap / nwindows)
            
            valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
            push!(valid_X, collect(SoleBase.movingaverage.(Array(X[i, :]); nwindows = nwindows, relative_overlap = overlap) for i in 1:nrow(X))...)

        # elseif model.data_treatment == :reduce_aggregate
        #     valid_X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")w", k) for j in features for i in vnames for k in 1:nwindows]])
        #     win_X = [vcat(SoleBase.movingwindow.(Array(row); nwindows = nwindows, relative_overlap = overlap)...) for row in eachrow(X)]
        #     push!(valid_X, [vcat([map(f, i) for f in features for i in row]...) for row in eachrow(win_X)]...)
        end
    else
        # -------------------------------------------------------------------- #
        #                                worlds                                #
        # -------------------------------------------------------------------- #
        maxsize = argmax(length.(Array(X[!, :])))
        intervals = collect(allworlds(frame(X, maxsize.I[1])))

        valid_X = DataFrame([v => Vector{Float64}[] for v in [string(j, "(", i, ")") for j in features for i in vnames]])
        push!(valid_X, [vcat([map(f, Array(row[i])) for f in features for i in intervals]...) for row in eachrow(X)]...)
    end

    return valid_X
end
