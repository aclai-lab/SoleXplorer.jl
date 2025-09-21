# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Saveable = Union{
    AbstractDataSet,
    AbstractSoleModel,
    AbstractModelSet
}

function _save(
    item    :: Saveable,
    prename :: AbstractString;
    path    :: AbstractString=@__DIR__,
    name    :: AbstractString
)::Nothing
    # check if name and path
    endswith(name, ".jld2") || (name = name * ".jld2")
    endswith(path, "/")     || (path = path * "/")
    filepath = joinpath(path, "$(prename)_$(name)")
    
    # save the item using JLD2
    jldsave(filepath; item=item)
    
    println("Saved $(typeof(item)) to: $filepath")
    return nothing
end

solesave(ds::AbstractDataSet; kwargs...)   = _save(ds, "sole_ds"; kwargs...)
solesave(ds::AbstractSoleModel; kwargs...) = _save(ds, "sole_model"; kwargs...)
solesave(ds::AbstractModelSet; kwargs...)  = _save(ds, "sole_analysis"; kwargs...)

function soleload()

end