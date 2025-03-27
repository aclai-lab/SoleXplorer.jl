# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
function fitmodel(modelset::AbstractModelSet, classifier::MLJ.Model, ds::Dataset; kwargs...)
    Xtrain, ytrain = ds.Xtrain isa AbstractDataFrame ? ([ds.Xtrain], [ds.ytrain]) : (ds.Xtrain, ds.ytrain)
    # if haskey(modelset.params, :watchlist) && modelset.params.watchlist == makewatchlist
    #     modelset.params = merge(modelset.params, (watchlist = makewatchlist(ds.Xtrain, ds.ytrain, ds.Xvalid, ds.yvalid),))
    # end

    mach = [MLJ.machine(classifier, x, y; kwargs...) |> m -> fit!(m, verbosity=0) for (x, y) in zip(Xtrain, ytrain)]

    return length(mach) == 1 ? only(mach) : mach
end

# ---------------------------------------------------------------------------- #
#                                  test model                                  #
# ---------------------------------------------------------------------------- #
function testmodel(modelset::AbstractModelSet, mach::Union{T, AbstractVector{<:T}}, ds::Dataset) where T<:MLJ.Machine
    mach isa AbstractVector || (mach = [mach])
    Xtest, ytest = ds.Xtest isa AbstractDataFrame ? ([ds.Xtest], [ds.ytest]) : (ds.Xtest, ds.ytest)
    tmodel = [modelset.learn_method(m, x, y) for (m, x, y) in zip(mach, Xtest, ytest)]

    return length(tmodel) == 1 ? only(tmodel) : tmodel
end
