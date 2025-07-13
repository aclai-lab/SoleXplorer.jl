# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractModelSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# const DataSetType = Union{
#     PropositionalDataSet{<:MLJ.Model},
#     ModalDataSet{<:Modal},
# }

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
get_X(model::AbstractDataSet)::DataFrame = model.mach.args[1].data
get_y(model::AbstractDataSet)::Vector = model.mach.args[2].data

# ---------------------------------------------------------------------------- #
#                                   modelset                                   #
# ---------------------------------------------------------------------------- #
mutable struct ModelSet{D} <: AbstractModelSet
    sole   :: Vector{AbstractModel}

    function ModelSet(::D, sole::Vector{AbstractModel}) where D<:AbstractDataSet
        new{D}(sole)
    end
end

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _train_test(model::AbstractDataSet)
    n_folds     = length(model.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train, test = get_train(model.pidxs[i]), get_test(model.pidxs[i])
        X_test, y_test = get_X(model)[test, :], get_y(model)[test]

        MLJ.fit!(model.mach, rows=train, verbosity=0)
        solemodel[i] = apply(model, X_test, y_test)
    end

    return ModelSet(model, solemodel)
end

function train_test(args...; kwargs...)
    model = _prepare_dataset(args...; kwargs...)
    _train_test(model)
end

train_test(model::AbstractDataSet) = _train_test(model)
