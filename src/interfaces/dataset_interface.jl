# ---------------------------------------------------------------------------- #
#                                  dataset info                                #
# ---------------------------------------------------------------------------- #
struct DatasetInfo <: AbstractDatasetInfo
    treatment   :: Symbol
    modalreduce :: Base.Callable
    train_ratio :: Real
    valid_ratio :: Real
    rng         :: AbstractRNG
    vnames      :: Vector{Symbol}

    function DatasetInfo(
        treatment   :: Symbol,
        modalreduce :: Base.Callable,
        train_ratio :: Real,
        valid_ratio :: Real,
        rng         :: AbstractRNG,
        vnames      :: Vector{Symbol}
    ) :: DatasetInfo
        # Validate ratios
        0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
        0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

        new(treatment, modalreduce, train_ratio, 1-valid_ratio, rng, vnames)
    end
end

function Base.show(io::IO, info::DatasetInfo)
    println(io, "DatasetInfo:")
    for field in fieldnames(DatasetInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

# ---------------------------------------------------------------------------- #
#                             datasplit collection                             #
# ---------------------------------------------------------------------------- #
struct DataSplit{T<:Integer} <: AbstractDataSplit
    train :: Vector{T}
    valid :: Vector{T}
    test  :: Vector{T}

    function DataSplit(
        train :: Vector{T},
        args...
    ) :: DataSplit{T} where {T<:Integer}
        new{T}(train, args...)
    end
end

# ---------------------------------------------------------------------------- #
#                                   dataset                                    #
# ---------------------------------------------------------------------------- #
struct Dataset{T,S} <: AbstractDataset
    X           :: Matrix{T}
    y           :: Vector{S}
    tt          :: Vector{<:DataSplit}
    info        :: DatasetInfo

    function Dataset(
        X       :: Matrix{T},
        y       :: AbstractVector{S},
        args...
    ) :: Dataset{T,S} where {T,S}
        # validate input dimensions
        size(X, 1) == length(y) || throw(ArgumentError("Number of rows in X must match length of y"))

        new{T,S}(X, y isa Vector ? y : Vector(y), args...)
    end
end

get_X(     ds :: Dataset) :: Matrix{T} where T    = ds.X
get_y(     ds :: Dataset) :: Vector{S} where S    = ds.y
get_tt(    ds :: Dataset) :: Vector{<:DataSplit} = ds.tt
get_info(  ds :: Dataset) :: DatasetInfo          = ds.info

# tt structure
get_train( ds :: Dataset) :: Vector{Vector{Integer}} = collect(x.train for x in ds.tt)
get_valid( ds :: Dataset) :: Vector{Vector{Integer}} = collect(x.valid for x in ds.tt)
get_test(  ds :: Dataset) :: Vector{Vector{Integer}} = collect(x.test  for x in ds.tt)

get_train( ds :: Dataset, i :: Integer) :: Vector{<:Integer} = ds.tt[i].train
get_valid( ds :: Dataset, i :: Integer) :: Vector{<:Integer} = ds.tt[i].valid
get_test(  ds :: Dataset, i :: Integer) :: Vector{<:Integer} = ds.tt[i].test 

# info structure
get_treatment(   ds :: Dataset) :: Symbol           = ds.info.treatment
get_modalreduce( ds :: Dataset) :: Base.Callable    = ds.info.modalreduce
get_train_ratio( ds :: Dataset) :: Real             = ds.info.train_ratio
get_valid_ratio( ds :: Dataset) :: Real             = ds.info.valid_ratio
get_rng(         ds :: Dataset) :: AbstractRNG      = ds.info.rng
get_vnames(      ds :: Dataset) :: Vector{Symbol} = ds.info.vnames

function Base.show(io::IO, ds::Dataset)
    println(io, "Dataset:")
    println(io, "  X shape:        ",       size(  ds.X))
    println(io, "  y length:       ",       length(ds.y))
    println(io, "  Train/Valid/Test:     ", length(ds.tt), " folds")
    println(io, get_info(ds))
end