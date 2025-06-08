# ---------------------------------------------------------------------------- #
#                              Results structs                                 #
# ---------------------------------------------------------------------------- #
struct ClassResults <: AbstractResults
    accuracy   :: AbstractFloat
    # rules      :: Union{Rule,          Nothing}
    # metrics    :: Union{AbstractVector, Nothing}
    # feature_importance :: Union{AbstractVector, Nothing}
    # predictions:: Union{AbstractVector, Nothing}
end

# function Base.show(io::IO, ::MIME"text/plain", r::ClassResults)
#     println(io, "Results")
#     println(io, "  Accuracy: ", r.accuracy)
#     println(io, "  Rules: ", r.rules)
#     println(io, "  Feature importance: ", r.feature_importance)
#     println(io, "  Predictions: ", r.predictions)
# end

# Base.show(io::IO, r::ClassResults) = print(io, "Results(accuracy=$(r.accuracy), rules=$(r.rules))")

struct RegResults <: AbstractResults
    accuracy   :: AbstractFloat
    # rules      :: Union{Rule,          Nothing}
    # metrics    :: Union{AbstractVector, Nothing}
    # feature_importance :: Union{AbstractVector, Nothing}
    # predictions:: Union{AbstractVector, Nothing}
end

const RESULTS = Dict{Symbol,DataType}(
    :classification => ClassResults,
    :regression     => RegResults
)