using Documenter
using SoleXplorer

# DocMeta.setdocmeta!(SoleXplorer, :DocTestSetup, :(using SoleXplorer); recursive = true)

# makedocs(;
#     modules=[SoleXplorer],
#     authors=["Lorenzo BALBONI", "Mauro MILELLA", "Giovanni PAGLIARINI", "Alberto PAPARELLA", "Riccardo PASINI", "Marco PERROTTA"],
#     repo=Documenter.Remotes.GitHub("aclai-lab", "SoleXplorer.jl"),
#     sitename="SoleXplorer.jl",
#     format=Documenter.HTML(;
#         size_threshold=4000000,
#         prettyurls=get(ENV, "CI", "false") == "true",
#         canonical="https://aclai-lab.github.io/SoleXplorer.jl",
#         assets=String[],
#     ),
#     pages=[
#         "Home"      => "index.md",
#         "Reference" => "reference.md",
#     ],
#     warnonly=:true,
# )

# deploydocs(;
#     repo = "github.com/aclai-lab/SoleXplorer.jl",
#     devbranch = "main",
#     target = "build",
#     branch = "gh-pages",
#     versions = ["main" => "main", "stable" => "v^", "v#.#"],
# )
