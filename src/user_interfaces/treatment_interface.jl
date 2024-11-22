# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
function is_valid_type(X::AbstractDataFrame, valid_type::Type = AbstractFloat)
    column_eltypes = eltype.(eachcol(X))
    is_valid = map(t -> t <: valid_type || (t <: AbstractArray && eltype(t) <: valid_type), column_eltypes)
    any(x -> x == 1, is_valid)
end

hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

function check_vector_dataframe(X::AbstractDataFrame)
    column_types = eltype.(eachcol(X))
    vector_lengths = length.(collect(eachcol(X)))
    is_vector = all(t -> t <: AbstractVector, column_types)
    
    is_vector ? all(==(vector_lengths[1]), vector_lengths) : false
end

# ---------------------------------------------------------------------------- #
#                               get treatment                                  #
# ---------------------------------------------------------------------------- #

function get_treatment(X::DataFrame, model::T, features::AbstractVector; 
    vnames::Union{AbstractVector{Union{String, Symbol}}, Nothing}=nothing,
    treatment::Union{Base.Callable, Nothing}=nothing,
    # nwindows::Union{Int, AbstractFloat, Nothing}=nothing,
    # overlap::Union{Int, AbstractFloat, Nothing}=nothing,
    kwargs...
) where {T<:SoleXplorer.ModelConfig}
    # ------------------------------------------------------------------------ #
    #                           check dataframe                                #
    # ------------------------------------------------------------------------ #
    is_valid_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))

    hasnans(X) && @warn "DataFrame contains NaN values"
    check_vector_dataframe(X) || @warn "DataFrame contains vectors of different lengths"

    # isnothing(nwindows) && (nwindows = 10)
    # isnothing(overlap) && (overlap = 0.2)

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
            # overlap isa Int && (overlap = overlap / nwindows)
            
            valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
            push!(valid_X, collect(SoleBase.movingaverage.(Array(X[i, :]); kwargs...) for i in 1:nrow(X))...)

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
        valid_intervals = treatment(intervals; kwargs...)

        valid_X = DataFrame([v => Float64[] for v in [string(f, "(", v, ")w", i) for f in features for v in vnames for (i, _) in enumerate(valid_intervals)]])

        win_X = Vector{Vector{Float64}}[]
        for row in eachrow(X)    # for vec in row
                push!(win_X, [vcat(vec[i.x:i.y-1]) for vec in row for i in valid_intervals])
        end

        push!(valid_X, [vcat([map(f, i) for f in features for i in row]...) for row in eachrow(win_X)]...)
    end

    return valid_X
end
