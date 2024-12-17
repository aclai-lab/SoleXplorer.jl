# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
function modelfit!(
    model::T,
    ds::S;
    features::Union{Function, AbstractVector}=catch9,
    fixcallablenans = false,
    kwargs...
) where {T<:SoleXplorer.ModelConfig, S<:SoleXplorer.Dataset}
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
