#!/usr/bin/env julia
import Pkg
Pkg.activate(@__DIR__ * "/..")
Pkg.instantiate()

using SemioticTransformer

seed = parse(Int, get(ENV, "SEMIOTIC_SEED", "2025"))
@info "Launching toy_train" seed
SemioticTransformer.toy_train(; seed=seed)
