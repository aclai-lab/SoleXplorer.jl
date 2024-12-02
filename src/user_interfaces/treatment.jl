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
    kwargs...
) where {T<:SoleXplorer.ModelConfig}
    # ------------------------------------------------------------------------ #
    #                           check dataframe                                #
    # ------------------------------------------------------------------------ #
    is_valid_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    hasnans(X) && @warn "DataFrame contains NaN values"
    check_vector_dataframe(X) || @warn "DataFrame contains vectors of different lengths"

    # ------------------------------------------------------------------------ #
    #                        check sub features names                          #
    # ------------------------------------------------------------------------ #
    if !isnothing(vnames)
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames = eltype(vnames) <: Symbol ? string.(vnames) : vnames
    else
        vnames = names(X)
    end

    # ------------------------------------------------------------------------ #
    #                                treatment                                 #
    # ------------------------------------------------------------------------ #
    isnothing(treatment) && (treatment = model.default_treatment; kwargs = (nwindows=10, overlap=0.3))

    maxsize_idx = argmax(length.(eachcol(X)))
    n_intervals = treatment(collect(allworlds(frame(X, maxsize_idx))); kwargs...)

    if model.data_treatment == :aggregate
        valid_X = DataFrame([v => Float64[] for v in [string(f, "(", v, ")w", i) for f in features for v in vnames for i in 1:length(n_intervals)]])

        for (n, row) in enumerate(eachrow(X))
            intervals = treatment(collect(allworlds(frame(X, n))); kwargs...)
            interval_diff = length(n_intervals) - length(intervals)
            push!(valid_X, vcat([vcat([f(col[i.x:i.y-1]) for i in intervals], fill(NaN, interval_diff)) for col in row, f in features]...))
        end

    elseif model.data_treatment == :reducesize
        valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])

        for (n, row) in enumerate(eachrow(X))
            intervals = treatment(collect(allworlds(frame(X, n))); kwargs...)
            interval_diff = length(n_intervals) - length(intervals)
            push!(valid_X, [vcat([mean(col[i.x:i.y-1]) for i in intervals], fill(NaN, interval_diff)) for col in row])
        end
    end

    return valid_X
end
