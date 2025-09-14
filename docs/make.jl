using Documenter
using SoleXplorer

# DocMeta.setdocmeta!(SoleXplorer, :DocTestSetup, :(using SoleXplorer); recursive = true)

makedocs(;
    modules=[SoleXplorer],
    authors="Lorenzo Balboni, Mauro Milella, Giovanni Pagliarini, Alberto Paparella, Riccardo Pasini, Marco Perrotta",
    repo=Documenter.Remotes.GitHub("aclai-lab", "SoleXplorer.jl"),
    sitename="SoleXplorer.jl",
    format=Documenter.HTML(;
        size_threshold=4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aclai-lab.github.io/SoleXplorer.jl",
        edit_link="main", # possibly this line is dangerous after publishing
        assets=String[],
    ),
    pages=[
        "Home"      => "index.md",
        # "Reference" => "reference.md",
    ],
    warnonly=:true,
)

# deploydocs(;
#     repo = "github.com/aclai-lab/SoleXplorer.jl",
#     devbranch = "main",
#     target = "build",
#     branch = "gh-pages",
#     versions = ["main" => "main", "stable" => "v^", "v#.#"],
# )
