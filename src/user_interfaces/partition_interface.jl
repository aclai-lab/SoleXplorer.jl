# ---------------------------------------------------------------------------- #
#                                data structs                                  #
# ---------------------------------------------------------------------------- #
struct TTIdx
    train::AbstractVector{<:Int}
    test::AbstractVector{<:Int}
end

# ---------------------------------------------------------------------------- #
#                                 partitioning                                 #
# ---------------------------------------------------------------------------- #
function get_partition(y::CategoricalArray; 
    stratified_sampling::Bool=false,
    train_ratio::Float64=0.7,
    nfolds::Int64=6,
    shuffle::Bool=true,
    rng::AbstractRNG=Random.TaskLocalRNG()
)
    if stratified_sampling
        stratified_cv = StratifiedCV(; nfolds=nfolds, shuffle=shuffle, rng=rng)
        tt_pairs = MLJ.MLJBase.train_test_pairs(stratified_cv, 1:length(y), y)
        return [TTIdx(train, test) for (train, test) in tt_pairs]
    else
        return [TTIdx(MLJ.partition(eachindex(y), train_ratio; shuffle=shuffle, rng=rng)...)]
    end
end