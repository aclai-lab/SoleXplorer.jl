# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real, AbstractArray{<:Real}}, eachcol(df))

function get_fit(
    X::DataFrame,
    y::CategoricalArray,
    tt_pairs::AbstractVector{TTIdx},
    model::T;
    features::Union{Function, AbstractVector}=catch9,
    fixcallablenans = false,
    kwargs...
) where {T<:SoleXplorer.ModelConfig}
    # ------------------------------------------------------------------------ #
    #                         data check and treatment                         #
    # ------------------------------------------------------------------------ #
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only Real or Array{<:Real} columns"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

    valid_feats = features isa Function ? [features] : unique(vcat(features...))

    # ------------------------------------------------------------------------ #
    #                           train & fit model                              #
    # ------------------------------------------------------------------------ #
    fit_model = []
    for tt in tt_pairs
        mach = machine(model.classifier, X[tt.train, :], y[tt.train])
        fit!(mach, verbosity=0)
        push!(fit_model, mach)
    end

    return fit_model
end
