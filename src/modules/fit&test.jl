# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
function fitmodel(modelset::AbstractModelSet, classifier::MLJ.Model, ds::Dataset; kwargs...)

    Xtrain, ytrain = ds.Xtrain isa AbstractDataFrame ? ([ds.Xtrain], [ds.ytrain]) : (ds.Xtrain, ds.ytrain)

    # mach = MLJ.Machine[]
    # for i in eachindex(ytrain)
    #     fmodel = MLJ.machine(classifier, Xtrain[i], ytrain[i]; kwargs...) |> (m -> fit!(m, verbosity=0))
    #     push!(mach, fmodel)
    # end

    mach = [MLJ.machine(classifier, x, y; kwargs...) |> m -> fit!(m, verbosity=0) for (x, y) in zip(Xtrain, ytrain)]

    return length(mach) == 1 ? only(mach) : mach
end

# ---------------------------------------------------------------------------- #
#                                  test model                                  #
# ---------------------------------------------------------------------------- #
function testmodel(modelset::AbstractModelSet, mach::Union{T, AbstractVector{<:T}}, ds::Dataset) where T<:MLJ.Machine
    mach isa AbstractVector || (mach = [mach])
    Xtrain, ytrain = ds.Xtrain isa AbstractDataFrame ? ([ds.Xtrain], [ds.ytrain]) : (ds.Xtrain, ds.ytrain)
    tmodel = [modelset.learn_method(m, x, y) for (m, x, y) in zip(mach, Xtrain, ytrain)]

    return length(tmodel) == 1 ? only(tmodel) : tmodel
end
