# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real, AbstractArray{<:Real}}, eachcol(df))

function get_fit(
    model::T,
    X::DataFrame,
    y::CategoricalArray,
    tt_pairs::Union{TTIdx, AbstractVector{TTIdx}};
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
    valid_tt = tt_pairs isa TTIdx ? [tt_pairs] : tt_pairs

    # ------------------------------------------------------------------------ #
    #                           train & fit model                              #
    # ------------------------------------------------------------------------ #
    fitmodel = MLJ.Machine[]

    for tt in valid_tt
        mach = machine(model.classifier, selectrows(X, tt.train), y[tt.train])
        fit!(mach, verbosity=0)

        push!(fitmodel, mach)
    end

    return length(fitmodel) == 1 ? fitmodel[1] : fitmodel
end
