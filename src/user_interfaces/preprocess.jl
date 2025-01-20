# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
# function is_valid_type(X::AbstractDataFrame, valid_type::Type = AbstractFloat)
#     column_eltypes = eltype.(eachcol(X))
#     is_valid = map(t -> t <: valid_type || (t <: AbstractArray && eltype(t) <: valid_type), column_eltypes)
#     any(x -> x == 1, is_valid)
# end

check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real, AbstractArray{<:Real}}, eachcol(df))
hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

# function check_vector_dataframe(X::AbstractDataFrame)
#     column_types = eltype.(eachcol(X))
#     vector_lengths = length.(collect(eachcol(X)))
#     is_vector = all(t -> t <: AbstractVector, column_types)
    
#     is_vector ? all(==(vector_lengths[1]), vector_lengths) : false
# end

# ---------------------------------------------------------------------------- #
#                                 winparams                                    #
# ---------------------------------------------------------------------------- #
function _treatment(
    X::DataFrame, 
    model::AbstractModelSet, 
    vnames::AbstractVector{String};
    kwargs...
)
    max_interval = maximum(length.(eachrow(X)))
    n_intervals = model.winparams.type(max_interval; model.winparams.params...)
    
    if model.config.treatment == :aggregate        # propositional
        if n_intervals == 1
            valid_X = DataFrame([v => Float64[] 
                for v in [string(f, "(", v, ")") 
                for f in model.features for v in vnames]]
            )
        else
            valid_X = DataFrame([v => Float64[] 
                for v in [string(f, "(", v, ")w", i) 
                for f in model.features for v in vnames 
                for i in 1:length(n_intervals)]]
            )
        end

    elseif model.config.treatment == :reducesize   # modal
        valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
    else
        # TODO
        throw(ArgumentError("Column type not yet supported"))
    end

    for row in eachrow(X)
        row_intervals = model.winparams.type(maximum(length.(collect(row))); model.winparams.params...)
        interval_diff = length(n_intervals) - length(row_intervals)

        if model.config.treatment == :aggregate
            push!(valid_X, vcat([vcat([f(col[r]) for r in row_intervals], 
                                        fill(NaN, interval_diff)) 
                                        for col in row, f in model.features]...)
                                    )
        elseif model.config.treatment == :reducesize
            f = mean # TODO make it a parameter
            push!(valid_X, [vcat([f(col[r]) for r in row_intervals], 
                                    fill(NaN, interval_diff)) 
                                    for col in row
                                    ]
                                )
        end
    end

    return valid_X
end

# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
function _partition(y::Union{CategoricalArray, Vector{Float64}}; 
    stratified_sampling::Bool=false,
    train_ratio::Float64=0.7,
    nfolds::Int64=6,
    shuffle::Bool=true,
    rng::AbstractRNG=Random.TaskLocalRNG(),
    kwargs...
)
    if stratified_sampling
        stratified_cv = MLJ.StratifiedCV(; nfolds=nfolds, shuffle=shuffle, rng=rng)
        tt = MLJ.MLJBase.train_test_pairs(stratified_cv, 1:length(y), y)
        return [TT_indexes(train, test) for (train, test) in tt]
    else
        return TT_indexes(MLJ.partition(eachindex(y), train_ratio; shuffle=shuffle, rng=rng)...)
    end
end

# ---------------------------------------------------------------------------- #
#                             dataset preprocessing                            #
# ---------------------------------------------------------------------------- #
function preprocess_dataset(
    X::AbstractDataFrame, 
    y::AbstractVector, 
    model::AbstractModelSet;
    vnames::Union{AbstractVector{Union{String, Symbol}}, Nothing}=nothing,
    kwargs...
)
    # ------------------------------------------------------------------------ #
    #                           check parameters                               #
    # ------------------------------------------------------------------------ #
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

    if model.config.algo == :regression
        y isa AbstractFloat || (y = Float64.(y))
    else
        y isa CategoricalArray || (y = CategoricalArray(y))
    end

    hasnans(X) && @warn "DataFrame contains NaN values"
    # TODO nan handles

    if isnothing(vnames)
        vnames = names(X)
    else
        size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
        vnames = eltype(vnames) <: Symbol ? string.(vnames) : vnames
    end

    column_eltypes = eltype.(eachcol(X))

    if all(t -> t <: Number, column_eltypes)
        # dataframe with numeric columns
        SoleXplorer.Dataset(DataFrame(vnames .=> eachcol(X)), y, _partition(y; kwargs...))

    elseif all(t -> t <: AbstractVector{<:Number}, column_eltypes)
        # dataframe with vector-valued columns
        SoleXplorer.Dataset(_treatment(X, model, vnames; kwargs...), y, _partition(y; kwargs...))
    else
        # TODO
        throw(ArgumentError("Column type not yet supported"))
    end
end

function preprocess_dataset(
    X::AbstractDataFrame, 
    y::Union{Symbol, AbstractString}, 
    model::AbstractModelSet;
    kwargs...
)
    preprocess_dataset(X[!, Not(y)], X[!, y], model; kwargs...)
end
