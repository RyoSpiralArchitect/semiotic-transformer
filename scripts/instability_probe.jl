#!/usr/bin/env julia
import Pkg
Pkg.activate(@__DIR__ * "/..")
Pkg.instantiate()

using SemioticTransformer

seed = parse(Int, get(ENV, "SEMIOTIC_SEED", "2027"))
ε = parse(Float64, get(ENV, "SEMIOTIC_EPS", "5e-4"))
samples = parse(Int, get(ENV, "SEMIOTIC_SAMPLES", "4"))
λ_instab = parse(Float64, get(ENV, "SEMIOTIC_LAMBDA_INSTAB", "1e-2"))

@info "Running meaning-instability probe" seed ε samples λ_instab
probe = SemioticTransformer.instability_probe(; seed=seed, ε=Float32(ε), samples=samples, λ_instab=Float32(λ_instab))
@info probe
