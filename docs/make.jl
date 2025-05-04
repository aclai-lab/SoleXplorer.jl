using SoleXplorer
using Documenter

makedocs(;
    modules = [SoleXplorer],
    authors = "Riccardo Pasini",
    repo=Documenter.Remotes.GitHub("aclai-lab", "SoleXplorer.jl"),
    sitename = "SoleXplorer.jl",
    format = Documenter.HTML(;
        size_threshold = 4000000,
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/aclai-lab/SoleXplorer.jl",
        assets = String[],
    ),
    pages = [
        "Catch22 and featuresets" => "catch22_and_featuresets.md",
    ],
    # NOTE: warning
    warnonly = :true,
)