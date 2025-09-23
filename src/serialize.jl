# serialization functionality for SoleXplorer objects using the JLD2 format.
# It enables saving and loading of datasets, models, and analysis results with automatic file
# naming conventions and path management.

# All serialization functions work with types that implement the `Saveable` union:
# - AbstractDataSet: Dataset configurations and ML pipelines
# - AbstractSoleModel: Trained symbolic models  
# - AbstractModelSet: Complete analysis results with multiple models

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# union type defining all objects that can be serialized by SoleXplorer
const Saveable = Union{
    AbstractDataSet,
    AbstractSoleModel,
    AbstractModelSet
}

# ---------------------------------------------------------------------------- #
#                                 save model                                   #
# ---------------------------------------------------------------------------- #
# internal function for saving SoleXplorer objects to JLD2 format
function _save(
    sole    :: Saveable,
    prename :: AbstractString;
    path    :: AbstractString=@__DIR__,
    name    :: AbstractString
)::Nothing
    # check name and path
    endswith(name, ".jld2") || (name = name * ".jld2")
    endswith(path, "/")     || (path = path * "/")
    filepath = joinpath(path, "$(prename)_$(name)")

    # check if file exists
    isfile(filepath) && throw(ArgumentError("File already exists: $filepath."))
    
    # save the sole using JLD2
    jldsave(filepath; sole=sole)
    
    println("Saved $(typeof(sole)) to: $filepath")
    return nothing
end

"""
    solesave(sole::Saveable; path::AbstractString, name::AbstractString)

This function handles the core serialization logic including file path construction,
existence checking, and JLD2 serialization.

# Arguments
- `sole::Saveable`: Object to serialize
- `path::AbstractString`: Directory path for saving (defaults to current directory)
- `name::AbstractString`: Base filename (automatically gets .jld2 extension)

See also: [`soleload`](@ref)
"""
solesave(ds::AbstractDataSet; kwargs...)   = _save(ds, "soleds"; kwargs...)
solesave(ds::AbstractSoleModel; kwargs...) = _save(ds, "solemodel"; kwargs...)
solesave(ds::AbstractModelSet; kwargs...)  = _save(ds, "soleanalysis"; kwargs...)

# ---------------------------------------------------------------------------- #
#                                 load model                                   #
# ---------------------------------------------------------------------------- #
"""
    soleload(path::AbstractString, name::AbstractString)::Saveable

Load a previously saved SoleXplorer object from disk.

Restores datasets, models, or analysis results from JLD2 files created by
`solesave`. The returned object type depends on what was originally saved.

# Arguments
- `path::AbstractString`: Directory path containing the file
- `name::AbstractString`: Filename (with or without .jld2 extension)

# File Extension
Automatically adds `.jld2` extension if not provided in `name`.

See also: [`solesave`](@ref)
"""
function soleload(
    path :: AbstractString,
    name :: AbstractString
)::Saveable
    # check name and path
    endswith(name, ".jld2") || (name = name * ".jld2")
    endswith(path, "/")     || (path = path * "/")
    filepath = joinpath(path, "$(name)")

    # check if file exists
    isfile(filepath) || throw(ArgumentError("File doesn't exists: $filepath."))

    # load the sole using JLD2
    data = jldopen(filepath, "r") do file
        file["sole"]
    end
    
    println("Loaded $(typeof(data)) from: $filepath")
    return data
end