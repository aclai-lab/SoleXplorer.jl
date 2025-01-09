# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
function modelfit!(
    model::T,
    ds::S;
    features::Union{Function, AbstractVector}=catch9,
    kwargs... # TODO put the remaining kwargs into the MLJ model's kwargs?
) where {T<:SoleXplorer.ModelConfig,S<:SoleXplorer.Dataset}
    # ------------------------------------------------------------------------ #
    #                         data check and treatment                         #
    # ------------------------------------------------------------------------ #
    check_dataframe_type(ds.X) || throw(ArgumentError("DataFrame must contain only Real or Array{<:Real} columns"))
    size(ds.X, 1) == length(ds.y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

    valid_feats = features isa Function ? [features] : unique(vcat(features...))
    tt_train = ds.tt isa AbstractVector ? ds.tt : [ds.tt]

    fitmodel = MLJ.Machine[]

    for tt in tt_train
        # mach = if model.model_algo == :regression
        #     MLJ.machine(model.classifier, selectrows(ds.X, tt.train))
        # else
            mach = MLJ.machine(model.classifier, selectrows(ds.X, tt.train), ds.y[tt.train])
        # end
        fit!(mach, verbosity=0)

        push!(fitmodel, mach)
    end

    model.mach = length(fitmodel) == 1 ? fitmodel[1] : fitmodel
end
