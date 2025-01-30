# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
function fitmodel(
    modelset::AbstractModelSet,
    classifier::MLJ.Model,
    ds::Dataset;
    kwargs...
)
    if ds.Xtrain isa AbstractDataFrame
        Xtrain, ytrain = [ds.Xtrain], [ds.ytrain]
    else
        Xtrain, ytrain = ds.Xtrain, ds.ytrain
    end

    mach = MLJ.Machine[]

    for i in eachindex(ytrain)
        fmodel = if modelset.config.algo == :regression
            MLJ.machine(classifier, Xtrain[i]; kwargs...)
        elseif modelset.config.algo == :classification
            fmodel = MLJ.machine(classifier, Xtrain[i], ytrain[i]; kwargs...)
        else
            throw(ArgumentError("Invalid algorithm type: $(modelset.config.algo)"))
        end

        fit!(fmodel, verbosity=0)
        push!(mach, fmodel)
    end

    return length(mach) == 1 ? first(mach) : mach
end

# ---------------------------------------------------------------------------- #
#                                  test model                                  #
# ---------------------------------------------------------------------------- #
function testmodel(
    modelset::AbstractModelSet,
    mach::MLJ.Machine,
    ds::Dataset
)
    if ds.Xtest isa AbstractDataFrame
        Xtest, ytest = [ds.Xtest], [ds.ytest]
    else
        Xtest, ytest = ds.Xtest, ds.ytest
    end

    tmodel = AbstractModel[]

    for i in eachindex(ytest)
        testset = modelset.learn_method(mach, Xtest[i], ytest[i])
        push!(tmodel, testset)
    end

    return length(tmodel) == 1 ? first(tmodel) : tmodel
end
