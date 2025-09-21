# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Saveable = Union{
    AbstractDataSet,
    AbstractSoleModel,
    AbstractModelSet
}

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

solesave(ds::AbstractDataSet; kwargs...)   = _save(ds, "soleds"; kwargs...)
solesave(ds::AbstractSoleModel; kwargs...) = _save(ds, "solemodel"; kwargs...)
solesave(ds::AbstractModelSet; kwargs...)  = _save(ds, "soleanalysis"; kwargs...)

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