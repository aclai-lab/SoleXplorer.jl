# questa struttura conserva tutte le informazioni necessarie per trattare un dataset multidimensionale.
# per ora, Sole accetta solo dataset numerici e, al più, serie temporali;
# ma è nostra intenzione estenderlo a qualsiasi dimensione di dataset.
# il fine ultimo è: 
# - per l'utilizzo dei normali algoritmi proposizionali, quali decision tree e xgboost,
# convertire il dataset multidimensionale, in un dataset numerico, suddividendo le serie temporali in 
# finestre, alle quali applicare una condizione.
# - per l'utilizzo di algoritmi modali, quali modaldecisiontree, 
# viene creata una struttura ad hoc: sole logiset.
# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTreatmentInfo end

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
mutable struct TreatmentInfo <: AbstractTreatmentInfo
    features    :: Union{Vector{<:Base.Callable}, Nothing},
    winparams   :: WinParams,
    treatment   :: Symbol=:aggregate,
    modalreduce :: OptCallable=nothing
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function treatment end

function treatment(
    X           :: AbstractDataFrame;
    features    :: Union{Vector{<:Base.Callable}, Nothing},
    winparams   :: WinParams,
    treat       :: Symbol=:aggregate,
    modalreduce :: OptCallable=nothing
)
    vnames = propertynames(X)

    # working with audio files, we need to consider audio of different lengths.
    # max_interval = first(find_max_length(X))
    # n_intervals = winparams.type(max_interval; winparams.params...)

    # define column names and prepare data structure based on treatment type
    if treat == :aggregate        # propositional
        if length(n_intervals) == 1
            col_names = [string(f, "(", v, ")") for f in features for v in vnames]
            
            n_rows = size(X, 1)
            n_cols = length(col_names)
            result_matrix = Matrix{eltype(T)}(undef, n_rows, n_cols)
        else
            # define column names with features names and window indices
            col_names = [string(f, "(", v, ")w", i) 
                         for f in features 
                         for v in vnames 
                         for i in 1:length(n_intervals)]
            
            n_rows = size(X, 1)
            n_cols = length(col_names)
            result_matrix = Matrix{eltype(T)}(undef, n_rows, n_cols)
        end
            
        # fill matrix
        for (row_idx, row) in enumerate(eachrow(X))
            row_intervals = winparams.type(maximum(length.(collect(row))); winparams.params...)
            interval_diff = length(n_intervals) - length(row_intervals)

            # calculate feature values for this row
            feature_values = vcat([
                vcat([f(col[r]) for r in row_intervals],
                    fill(NaN, interval_diff)) for col in row, f in features
            ]...)
            result_matrix[row_idx, :] = feature_values
        end

    elseif treat == :reducesize   # modal
        col_names = vnames
        
        n_rows = size(X, 1)
        n_cols = length(col_names)
        result_matrix = Matrix{T}(undef, n_rows, n_cols)

        modalreduce === nothing && (modalreduce = mean)
        
        for (row_idx, row) in enumerate(eachrow(X))
            row_intervals = winparams.type(maximum(length.(collect(row))); winparams.params...)
            interval_diff = length(n_intervals) - length(row_intervals)
            
            # calculate reduced values for this row
            reduced_data = [
                vcat([modalreduce(col[r]) for r in row_intervals],
                     fill(NaN, interval_diff)) for col in row
            ]
            result_matrix[row_idx, :] = reduced_data
        end
    end

    return result_matrix, col_names
end