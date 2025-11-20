"""
Bootstrap the local project so `include("SemioticTransformer.jl")` works even when the
repository has not been activated yet. Set `SEMIOTIC_BOOTSTRAP=0` to skip automatic
activation/instantiation (for example, inside an already-active environment).
"""
if get(ENV, "SEMIOTIC_BOOTSTRAP", "1") != "0"
    import Pkg
    try
        Pkg.activate(@__DIR__)
        # Resolve first to repair any stale manifests before instantiating.
        Pkg.resolve()
        Pkg.instantiate()
    catch err
        @error "SemioticTransformer bootstrap failed; run `Pkg.resolve(); Pkg.instantiate()` manually" error=err
        rethrow(err)
    end
end

include(joinpath(@__DIR__, "src", "SemioticTransformer.jl"))
