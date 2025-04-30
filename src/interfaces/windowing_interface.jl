# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
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

const WIN_PARAMS = Dict(
    movingwindow   => (window_size = 1024, window_step = 512),
    wholewindow    => NamedTuple(),
    splitwindow    => (nwindows = 5,),
    adaptivewindow => (nwindows = 5, relative_overlap = 0.1)
)

# ---------------------------------------------------------------------------- #
#                                  InfoFeat                                    #
# ---------------------------------------------------------------------------- #
"""
    InfoFeat{T<:VarName} <: AbstractFeature

A structure that represents a feature extraction operation on a specific variable and window.

# Type Parameters
- `T`: Type of variable name (either `Symbol` or `String`)

# Fields
- `id :: Int`: Unique identifier for the feature
- `var :: T`: Variable name/identifier (column name in dataset)
- `feat :: Symbol`: Name of the feature extraction function
- `nwin :: Int`: Window number this feature applies to

# Constructor
```julia
InfoFeat(id::Int, var::VarName, feat::Symbol, nwin::Int)
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
