# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract type for feature struct
"""
abstract type AbstractFeature end

# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const VarName   = Union{Symbol, String}
const VarNames  = Union{Vector{String}, Vector{Symbol}, Nothing}
const FeatNames = Union{Vector{<:Base.Callable}, Nothing}

# ---------------------------------------------------------------------------- #
#                                data structures                               #
# ---------------------------------------------------------------------------- #
struct WinParams
    type         :: Base.Callable
    params       :: NamedTuple
end

const DEFAULT_FE = (features = catch9,)

const AVAIL_WINS       = (movingwindow, wholewindow, splitwindow, adaptivewindow)
const FE_AVAIL_WINS    = (wholewindow, splitwindow, adaptivewindow)
# const AVAIL_TREATMENTS = (:aggregate, :reducesize)

const WIN_PARAMS = Dict(
    movingwindow   => (window_size = 1024, window_step = 512),
    wholewindow    => NamedTuple(),
    splitwindow    => (nwindows = 20,),
    adaptivewindow => (nwindows = 20, relative_overlap = 0.5)
)

"""
    InfoFeat{T<:VarName} <: AbstractFeature

Holds information about dataset columns, used in feature selection.

# Type Parameters
- `T`: VarName type (must be either `Symbol` or `String`)

# Fields
- `id   :: Int`     : Unique identifier for the feature (Int or nothing)
- `feat :: Symbol`  : The feature extraction function name
- `var  :: T`       : The variable name/identifier
- `nwin :: Int`     : The window number (must be positive)

# Constructors
```julia
InfoFeat(id::Id, feat::Symbol, var::Union{Symbol,String}, nwin::Integer)
"""
struct InfoFeat{T<:VarName} <: AbstractFeature
    id     :: Int
    var    :: T
    feat   :: Symbol
    nwin   :: Int

    function InfoFeat(id::Int, var::VarName, feat::Symbol, nwin::Int)
        nwin > 0 || throw(ArgumentError("Window number must be positive"))
        new{typeof(var)}(id, var, feat, nwin)
    end
end

# Value access methods
Base.getproperty(f::InfoFeat, s::Symbol) = getfield(f, s)
Base.propertynames(::InfoFeat)           = (:id, :feat, :var, :nwin)

feature_id(f::InfoFeat)    = f.id
variable_name(f::InfoFeat) = f.var
feature_type(f::InfoFeat)  = f.feat
window_number(f::InfoFeat) = f.nwin

