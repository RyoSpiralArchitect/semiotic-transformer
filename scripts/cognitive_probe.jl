#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using SemioticTransformer
using Statistics

function envint(name, default)
    return get(ENV, name, string(default)) |> x -> try parse(Int, x) catch; default end
end
function envfloat(name, default)
    return get(ENV, name, string(default)) |> x -> try parse(Float64, x) catch; default end
end

seed = envint("SEMIOTIC_SEED", 0)
vocab = envint("SEMIOTIC_VOCAB", 16)
d = envint("SEMIOTIC_D", 24)
seq = envint("SEMIOTIC_SEQ", 8)
local_layers = envint("SEMIOTIC_LOCAL_LAYERS", 1)
local_k = envint("SEMIOTIC_LOCAL_K", 4)
local_z = envint("SEMIOTIC_LOCAL_Z", 12)
local_H = envint("SEMIOTIC_LOCAL_H", 3)
global_K = envint("SEMIOTIC_GLOBAL_K", 3)
global_ds = envint("SEMIOTIC_GLOBAL_DS", 12)
global_r = envint("SEMIOTIC_GLOBAL_R", 16)
global_λ_pair = envfloat("SEMIOTIC_GLOBAL_LAMBDA_PAIR", 0.5)
λ_global = envfloat("SEMIOTIC_LAMBDA_GLOBAL", 1.0)
λ_couple = envfloat("SEMIOTIC_LAMBDA_COUPLE", 1e-3)
λ_struct = envfloat("SEMIOTIC_LAMBDA_STRUCT", 1e-3)
λ_rules = envfloat("SEMIOTIC_LAMBDA_RULES", 1e-2)
λ_mono = envfloat("SEMIOTIC_LAMBDA_MONO", 1e-3)
λ_instab = envfloat("SEMIOTIC_LAMBDA_INSTAB", 0.0)
ε_instab = envfloat("SEMIOTIC_EPS_INSTAB", 1e-3)
instab_samples = envint("SEMIOTIC_INSTAB_SAMPLES", 1)

probe = SemioticTransformer.cognitive_probe(
    vocab=vocab,
    d=d,
    seq=seq,
    seed=seed,
    local_layers=local_layers,
    local_k=local_k,
    local_z=local_z,
    local_H=local_H,
    global_K=global_K,
    global_ds=global_ds,
    global_r=global_r,
    global_λ_pair=global_λ_pair,
    λ_global=λ_global,
    λ_couple=λ_couple,
    λ_struct=λ_struct,
    λ_rules=λ_rules,
    λ_mono=λ_mono,
    λ_instab=λ_instab,
    ε_instab=ε_instab,
    instab_samples=instab_samples,
)

println("tokens: " * join(probe.tokens, ", "))
println("Ltotal=$(probe.Ltotal)")
println("Lcouple=$(probe.Lcouple)")
println("local: $(probe.local)")
println("global: $(probe.global)")
println("psi snapshot potential mean: $(mean(probe.psi.Φ))")
