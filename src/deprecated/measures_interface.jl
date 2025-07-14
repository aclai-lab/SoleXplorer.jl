mutable struct Measures <: AbstractMeasures
    # yhat            :: AbstractVector
    per_fold        :: OptVector
    measures        :: OptVecMeas
    measures_values :: OptVector
    operations      :: OptVecCall

    # function Measures(
    #     yhat      :: AbstractVector,
    # )::Measures
    #     new(yhat, nothing, nothing, nothing, nothing)
    # end
end