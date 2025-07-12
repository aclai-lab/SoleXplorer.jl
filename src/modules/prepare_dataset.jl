# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Modal = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}

# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
# utilizzato in caso non venga specificato il modello da utilizzare
# restituisce un modello di classificazione o di regressione
# a seconda del tipo di y.
function _DefaultModel(y::AbstractVector)::MLJ.Model
    if     eltype(y) <: CLabel
        return DecisionTreeClassifier()
    elseif eltype(y) <: RLabel
        return DecisionTreeRegressor()
    else
        throw(ArgumentError("Unsupported type for y: $(eltype(y))"))
    end
end

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
function set_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    m.rng = rng
    return m
end

function set_conditions!(m::MLJ.Model, conditions::Vector{<:Base.Callable})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

# check_dataset_type(X::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real,AbstractArray{<:Real}}, eachcol(X))
# hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

# function check_row_consistency(X::AbstractMatrix) 
#     for row in eachrow(X)
#         # skip rows with only scalar values
#         any(el -> el isa AbstractArray, row) || continue
        
#         # find first array element to use as reference
#         ref_idx = findfirst(el -> el isa AbstractArray, row)
#         ref_idx === nothing && continue
        
#         ref_size = size(row[ref_idx])
        
#         # check if any array element has different size (short-circuit)
#         if any(row) do el
#                 el isa AbstractArray && size(el) != ref_size
#             end
#             return false
#         end
#     end
#     return true
# end

function code_dataset!(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = MLJ.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

function code_dataset!(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = MLJ.levelcode.(categorical(y)) 
    end
    
    return y
end

code_dataset!(X::AbstractDataFrame, y::AbstractVector) = code_dataset!(X), code_dataset!(y)

# function check_dimensions(X::AbstractMatrix)
#     isempty(X) && return 0
    
#     # Get reference dimensions from first element
#     first_col = first(eachcol(X))
#     ref_dims = ndims(first(first_col))
    
#     # Early dimension check
#     ref_dims > 1 && throw(ArgumentError("Elements more than 1D are not supported."))
    
#     # Check all columns maintain same dimensionality
#     all(col -> all(x -> ndims(x) == ref_dims, col), eachcol(X)) ||
#         throw(DimensionMismatch("Inconsistent dimensions across elements"))
    
#     return ref_dims
# end

# check_dimensions(df::DataFrame) = check_dimensions(Matrix(df))

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
mutable struct PropositionalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
end

mutable struct ModalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    tinfo   :: TreatmentInfo
end

function DataSet(
    mach    :: MLJ.Machine{M},
    pidxs   :: Vector{PartitionIdxs},
    pinfo   :: PartitionInfo;
    tinfo   :: Union{TreatmentInfo, Nothing} = nothing
) where {M<:MLJ.Model}
    isnothing(tinfo) ?
        PropositionalDataSet{M}(mach, pidxs, pinfo) :
        ModalDataSet{M}(mach, pidxs, pinfo, tinfo)
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function _prepare_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector;
    model         :: MLJ.Model               = _DefaultModel(y),
    resample      :: NamedTuple              = (type=Holdout(shuffle=true), train_ratio=0.7, rng=TaskLocalRNG()),
    win           :: WinFunction             = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Vector{<:Base.Callable} = [maximum, minimum],
    modalreduce   :: Base.Callable           = mean,
    tuning        :: NamedTupleBool          = false,
    # extract_rules :: NamedTupleBool = false,
    measures      :: OptTuple                = nothing,
)::AbstractDataSet
    # propagate user rng to every field that needs it
    rng = hasproperty(resample, :rng) ? resample.rng : TaskLocalRNG()
    # set rng if the model supports it
    hasproperty(model, :rng) && (model = set_rng!(model, rng))
    # ModalDecisionTrees package needs features to be passed in model params
    hasproperty(model, :features) && (model = set_conditions!(model, features))

    # questo if è relativo a dataset multidimensionali.
    # qui si decide come trattare tali dataset:
    # abbiamo 2 soluzioni: utilizzare i normali algoritmi di machine learning, che accettano
    # solo dataset numerici, oppure utilizzare logica modale.
    # nel primo caso i dataset verranno ridotti a dataset numerici,
    # applicando una feature (massimo, minimo, media, ...) su un numero definito di finestre.
    # nel secondo caso, per economia di calcolo, verranno ridotti per finestre,
    # secondo un parametro di riduzione 'modalreduce' tipicamente mean, comunque definito dall'utente.
    if X[1, 1] isa AbstractArray
        treat = model isa Modal ? :reducesize : :aggregate
        X, tinfo = treatment(X; win, features, treat, modalreduce)
    else
        X = code_dataset!(X)
        tinfo = nothing
    end

    mach = MLJ.machine(model, X, y)
    ttpairs, pinfo = partition(y; resample...)

    DataSet(mach, ttpairs, pinfo; tinfo)
end

prepare_dataset(args...; kwargs...) = _prepare_dataset(args...; kwargs...)

# y is not a vector, but a symbol that identifies a column in X
function prepare_dataset(
    X::AbstractDataFrame,
    y::Symbol;
    kwargs...
)::Tuple{Modelset, Dataset}
    prepare_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end
