using Documenter
using SoleXplorer

using SoleXplorer
using Documenter

makedocs(;
    modules = [SoleXplorer],
    authors = "Lorenzo Balboni, Giovanni Pagliarini, Riccardo Pasini",
    repo=Documenter.Remotes.GitHub("aclai-lab", "SoleXplorer.jl"),
    sitename = "SoleXplorer.jl",
    format = Documenter.HTML(;
        size_threshold = 4000000,
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/aclai-lab/SoleXplorer.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
    # NOTE: warning
    warnonly = :true,
)

@info "`makedocs` has finished running. "

deploydocs(;
    repo = "github.com/aclai-lab/SoleXplorer.jl",
    devbranch = "71-documentation",
    target = "build",
    branch = "gh-pages",
    # versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)

@info "`deploydocs` has finished running. "