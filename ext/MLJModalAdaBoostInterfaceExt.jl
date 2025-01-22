module  MLJModalAdaBoost

import MLJModelInterface
using MLJModelInterface.ScientificTypesBase

import SoleLogics: AbstractRelation
using SoleData
using SoleData.MLJUtils
using SoleData: TestOperator
import ModalDecisionTrees: InitialCondition, wrapdataset

# import DecisionTree
# import Tables
# using CategoricalArrays

# using SoleLogics
# using SoleLogics: AbstractRelation
# using SoleData
# using SoleData.MLJUtils
# using SoleData: TestOperator
# using SoleModels

using ModalDecisionTrees
using ModalDecisionTrees: InitialCondition

using Random

const MMI = MLJModelInterface
# const DT = DecisionTree
const PKG = "MLJModalAdaBoostInterface"

struct TreePrinter{T}
    tree::T
    features::Vector{Symbol}
end
(c::TreePrinter)(depth) = DT.print_tree(c.tree, depth, feature_names = c.features)
(c::TreePrinter)() = DT.print_tree(c.tree, 5, feature_names = c.features)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")

function classes(y)
    p = CategoricalArrays.pool(y)
    [p[i] for i in 1:length(p)]
end





mutable struct ModalAdaBoost <: MMI.Probabilistic
    ## Pruning conditions
    max_depth               ::Union{Nothing,Int}
    min_samples_leaf        ::Union{Nothing,Int}
    min_purity_increase     ::Union{Nothing,Float64}
    max_purity_at_leaf      ::Union{Nothing,Float64}
    max_modal_depth         ::Union{Nothing,Int}

    ## Logic parameters

    # Relation set
    relations               ::Union{
        Nothing,                                            # defaults to a well-known relation set, depending on the data;
        Symbol,                                             # one of the relation sets specified in AVAILABLE_RELATIONS;
        Vector{<:AbstractRelation},                         # explicitly specify the relation set;
        # Vector{<:Union{Symbol,Vector{<:AbstractRelation}}}, # MULTIMODAL CASE: specify a relation set for each modality;
        Function                                            # A function worldtype -> relation set.
    }

    # Condition set
    features                ::Union{
        Nothing,                                                                   # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                    # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
    }
    conditions              ::Union{
        Nothing,                                                                   # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                    # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
    }
    # Type for the extracted feature values
    featvaltype             ::Type

    # Initial conditions
    initconditions          ::Union{
        Nothing,                                                                   # defaults to standard conditions (e.g., start_without_world)
        Symbol,                                                                    # one of the initial conditions specified in AVAILABLE_INITIALCONDITIONS;
        InitialCondition,                                                          # explicitly specify an initial condition for the learning algorithm.
    }

    ## Miscellaneous
    downsize                ::Union{Bool,NTuple{N,Integer} where N,Function}
    force_i_variables       ::Bool
    fixcallablenans         ::Bool
    print_progress          ::Bool
    
    ## DecisionTree.jl parameters
    display_depth           ::Union{Nothing,Int}
    min_samples_split       ::Union{Nothing,Int}
    n_subfeatures           ::Union{Nothing,Int,Float64,Function}
    post_prune              ::Bool
    merge_purity_threshold  ::Union{Nothing,Float64}
    
    ## AdaBoost parameters
    n_iter                  ::Int
    feature_importance      ::Symbol
    rng                     ::Union{Random.AbstractRNG,Integer}
end

function ModalAdaBoost(;
    max_depth = 1,
    min_samples_leaf = 4,
    min_purity_increase = 0.002,
    max_purity_at_leaf = Inf,
    max_modal_depth = nothing,
    #
    relations = :IA7,
    features = nothing,
    conditions = nothing,
    featvaltype = Float64,
    initconditions = nothing,
    #
    downsize = true,
    force_i_variables = true,
    fixcallablenans = true,
    print_progress = false,
    
    #
    display_depth = nothing,
    min_samples_split = nothing,
    n_subfeatures = identity,
    post_prune = false,
    merge_purity_threshold = nothing,
    
    #
    n_iter = 10,
    feature_importance = :impurity,
    rng = Random.GLOBAL_RNG,
)
    model = ModalAdaBoost(
        max_depth,
        min_samples_leaf,
        min_purity_increase,
        max_purity_at_leaf,
        max_modal_depth,
        #
        relations,
        features,
        conditions,
        featvaltype,
        initconditions,
        #
        downsize,
        force_i_variables,
        fixcallablenans,
        print_progress,
        #
        display_depth,
        min_samples_split,
        n_subfeatures,
        post_prune,
        merge_purity_threshold,
        #
        n_iter,
        feature_importance,
        rng
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

using SoleData.DimensionalDatasets

const ALLOW_GLOBAL_SPLITS = true

const mlj_default_max_depth = nothing
const mlj_default_max_modal_depth = nothing

const mlj_mdt_default_min_samples_leaf = 4
const mlj_mdt_default_min_purity_increase = 0.002
const mlj_mdt_default_max_purity_at_leaf = Inf
const mlj_mdt_default_n_subfeatures = identity

const mlj_mrf_default_min_samples_leaf = 1
const mlj_mrf_default_min_purity_increase = -Inf
const mlj_mrf_default_max_purity_at_leaf = Inf
const mlj_mrf_default_ntrees = 50
sqrt_f(x) = ceil(Int, sqrt(x))
const mlj_mrf_default_n_subfeatures = sqrt_f
const mlj_mrf_default_sampling_fraction = 0.7

mlj_default_initconditions = nothing

mlj_default_initconditions_str = "" *
    ":start_with_global" # (i.e., starting with a global decision, such as ⟨G⟩ min(V1) > 2) " *
    # "for 1-dimensional data and :start_at_center for 2-dimensional data."

AVAILABLE_INITCONDITIONS = OrderedDict{Symbol,InitialCondition}([
    :start_with_global => MDT.start_without_world,
    :start_at_center   => MDT.start_at_center,
])


function readinitconditions(model, dataset)
    if SoleData.ismultilogiseed(dataset)
        map(mod->readinitconditions(model, mod), eachmodality(dataset))
    else
        if model.initconditions == mlj_default_initconditions
            # d = dimensionality(SoleData.base(dataset)) # ? TODO maybe remove base for AbstractModalLogiset's?
            d = dimensionality(frame(dataset, 1))
            if d == 0
                AVAILABLE_INITCONDITIONS[:start_with_global]
            elseif d == 1
                AVAILABLE_INITCONDITIONS[:start_with_global]
            elseif d == 2
                AVAILABLE_INITCONDITIONS[:start_with_global]
            else
                error("Unexpected dimensionality: $(d)")
            end
        else
            AVAILABLE_INITCONDITIONS[model.initconditions]
        end
    end
end

function get_kwargs(m::ModalAdaBoost, X)
    base_kwargs = (;
        loss_function             = nothing,
        max_depth                 = m.max_depth,
        min_samples_leaf          = m.min_samples_leaf,
        min_purity_increase       = m.min_purity_increase,
        max_purity_at_leaf        = m.max_purity_at_leaf,
        max_modal_depth           = m.max_modal_depth,
        ####################################################################################
        n_subrelations            = identity,
        n_subfeatures             = m.n_subfeatures,
        initconditions            = readinitconditions(m, X),
        allow_global_splits       = ALLOW_GLOBAL_SPLITS,
        ####################################################################################
        use_minification          = false,
        perform_consistency_check = false,
        ####################################################################################
        rng                       = m.rng,
        print_progress            = m.print_progress,
    )

    additional_kwargs = (; n_iter = m.n_iter)
    merge(base_kwargs, additional_kwargs)
end

function MMI.clean!(m::ModalAdaBoost)
    warning = ""
    mlj_default_min_samples_leaf = mlj_mdt_default_min_samples_leaf
    mlj_default_min_purity_increase = mlj_mdt_default_min_purity_increase
    mlj_default_max_purity_at_leaf = mlj_mdt_default_max_purity_at_leaf
    mlj_default_n_subfeatures = mlj_mdt_default_n_subfeatures
    # TODO mlj_default_n_iter

    if !(isnothing(m.max_depth) || m.max_depth ≥ -1)
        warning *= "max_depth must be ≥ -1, but $(m.max_depth) " *
            "was provided. Defaulting to $(mlj_default_max_depth).\n"
        m.max_depth = mlj_default_max_depth
    end

    if !(isnothing(m.min_samples_leaf) || m.min_samples_leaf ≥ 1)
        warning *= "min_samples_leaf must be ≥ 1, but $(m.min_samples_leaf) " *
            "was provided. Defaulting to $(mlj_default_min_samples_leaf).\n"
        m.min_samples_leaf = mlj_default_min_samples_leaf
    end

    if !(isnothing(m.max_modal_depth) || m.max_modal_depth ≥ -1)
        warning *= "max_modal_depth must be ≥ -1, but $(m.max_modal_depth) " *
            "was provided. Defaulting to $(mlj_default_max_modal_depth).\n"
        m.max_modal_depth = mlj_default_max_depth
    end

    # Patch parameters: -1 -> nothing
    m.max_depth == -1 && (m.max_depth = nothing)
    m.max_modal_depth == -1 && (m.max_modal_depth = nothing)
    m.display_depth == -1 && (m.display_depth = nothing)

    # Patch parameters: nothing -> default value
    isnothing(m.max_depth)           && (m.max_depth           = mlj_default_max_depth)
    isnothing(m.min_samples_leaf)    && (m.min_samples_leaf    = mlj_default_min_samples_leaf)
    isnothing(m.min_purity_increase) && (m.min_purity_increase = mlj_default_min_purity_increase)
    isnothing(m.max_purity_at_leaf)  && (m.max_purity_at_leaf  = mlj_default_max_purity_at_leaf)
    isnothing(m.max_modal_depth)     && (m.max_modal_depth     = mlj_default_max_modal_depth)

    # Patch name: features -> conditions
    if !isnothing(m.features)
        if !isnothing(m.conditions)
            error("Please, only specify one hyper-parameter in `features` and `conditions`." *
                "Given: features = $(m.features) & conditions = $(m.conditions).")
        end
        m.conditions = m.features
        m.features = nothing
    end
    
    m.relations, _w = SoleData.autorelations(m.relations); warning *= _w
    m.conditions, _w = SoleData.autoconditions(m.conditions); warning *= _w
    m.downsize, _w = SoleData.autodownsize(m); warning *= _w

    if !(isnothing(m.initconditions) ||
        m.initconditions isa Symbol && m.initconditions in keys(AVAILABLE_INITCONDITIONS) ||
        m.initconditions isa InitialCondition
    )
        warning *= "initconditions should be in $(collect(keys(AVAILABLE_INITCONDITIONS))), " *
            "but $(m.initconditions) " *
            "was provided. Defaulting to $(mlj_default_initconditions_str).\n"
        m.initconditions = nothing
    end

    isnothing(m.initconditions) && (m.initconditions  = mlj_default_initconditions)

    if m.rng isa Integer
        m.rng = Random.MersenneTwister(m.rng)
    end

    if !(isnothing(m.min_samples_split) || m.min_samples_split ≥ 2)
        warning *= "min_samples_split must be ≥ 2, but $(m.min_samples_split) " *
            "was provided. Defaulting to $(nothing).\n"
        m.min_samples_split = nothing
    end

    if !isnothing(m.min_samples_split)
        m.min_samples_leaf = max(m.min_samples_leaf, div(m.min_samples_split, 2))
    end

    if m.n_subfeatures isa Integer && !(m.n_subfeatures > 0)
        warning *= "n_subfeatures must be > 0, but $(m.n_subfeatures) " *
            "was provided. Defaulting to $(nothing).\n"
        m.n_subfeatures = nothing
    end

    # Legacy behaviour
    m.n_subfeatures == -1 && (m.n_subfeatures = sqrt_f)
    m.n_subfeatures == 0 && (m.n_subfeatures = identity)

    function make_n_subfeatures_function(n_subfeatures)
        if isnothing(n_subfeatures)
            mlj_default_n_subfeatures
        elseif n_subfeatures isa Integer
            warning *= "An absolute n_subfeatures was provided $(n_subfeatures). " *
                "It is recommended to use relative values (between 0 and 1), interpreted " *
                "as the share of the random portion of feature space explored at each split."
            x -> convert(Int64, n_subfeatures)
        elseif n_subfeatures isa AbstractFloat
            @assert 0 ≤ n_subfeatures ≤ 1 "Unexpected value for " *
                "n_subfeatures: $(n_subfeatures). It should be ∈ [0,1]"
            x -> ceil(Int64, x*n_subfeatures)
        elseif n_subfeatures isa Function
            # x -> ceil(Int64, n_subfeatures(x)) # Generates too much nesting
            n_subfeatures
        else
            error("Unexpected value for n_subfeatures: $(n_subfeatures) " *
                "(type: $(typeof(n_subfeatures)))")
        end
    end

    m.n_subfeatures = make_n_subfeatures_function(m.n_subfeatures)

    if m.feature_importance == :impurity
        warning *= "feature_importance = :impurity is currently not supported." *
            "Defaulting to $(:split).\n"
        m.feature_importance == :split
    end

    if !(m.feature_importance in [:split])
        warning *= "feature_importance should be in [:split], " *
            "but $(m.feature_importance) " *
            "was provided.\n"
    end

    return warning
end

function MMI.fit(m::ModalAdaBoost, verbosity::Int, X, y, features, classes=nothing, w=nothing)

    integers_seen = unique(y)
    classes_seen  = MMI.decoder(classes)(integers_seen)

    stumps, coefs =
        DT.build_adaboost_stumps(y, X, m.n_iter, rng=m.rng)
    cache  = nothing

    report = (features=features,)

    return (stumps, coefs, classes_seen, integers_seen), cache, report
end

function MMI.reformat(m::ModalAdaBoost, X, y, w=nothing; passive_mode=false)
    @show typeof(X)
    X, var_grouping = wrapdataset(X, m; passive_mode)
    y, classes_seen = fix_y(y)
    (X, y, var_grouping, classes_seen, w)
end

MMI.selectrows(::ModalAdaBoost, I, X, y, var_grouping, classes_seen, w = nothing) =
    (MMI.selectrows(X, I), MMI.selectrows(y, I), var_grouping, classes_seen, MMI.selectrows(w, I),)

# For predict
function MMI.reformat(m::ModalAdaBoost, Xnew)
    Xnew, var_grouping = wrapdataset(Xnew, m; passive_mode = true)
    (Xnew, var_grouping)
end
MMI.selectrows(::ModalAdaBoost, I, Xnew, var_grouping) =
    (MMI.selectrows(Xnew, I), var_grouping,)

end