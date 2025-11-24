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
function envstr(name, default)
    return get(ENV, name, default)
end

seed = envint("SEMIOTIC_SEED", 0)
vocab = envint("SEMIOTIC_VOCAB", 16)
d = envint("SEMIOTIC_D", 24)
seq = envint("SEMIOTIC_SEQ", 8)
steps = envint("SEMIOTIC_TRACE_STEPS", 12)
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
λ_time = envfloat("SEMIOTIC_LAMBDA_TIME", 0.0)
λ_instab = envfloat("SEMIOTIC_LAMBDA_INSTAB", 0.0)
ε_instab = envfloat("SEMIOTIC_EPS_INSTAB", 1e-3)
instab_samples = envint("SEMIOTIC_INSTAB_SAMPLES", 1)
profile_path = envstr("SEMIOTIC_PROFILE_PATH", "")
coupling_path = envstr("SEMIOTIC_COUPLING_PATH", "")
trace_path = envstr("SEMIOTIC_TRACE_PATH", "")

profile_width = envint("SEMIOTIC_PROFILE_WIDTH", 28)

epsilons = begin
    raw = split(envstr("SEMIOTIC_EPS_LIST", ""), ",")
    vals = [try parse(Float64, r) catch; nothing end for r in raw]
    keep = filter(!isnothing, vals)
    isempty(keep) ? nothing : SemioticTransformer.T[keep...]
end

save_profile = isempty(profile_path) ? nothing : profile_path
save_coupling = isempty(coupling_path) ? nothing : coupling_path
save_trace = isempty(trace_path) ? nothing : trace_path

trace = SemioticTransformer.cognitive_trace(
    vocab=vocab,
    d=d,
    seq=seq,
    steps=steps,
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
    λ_time=λ_time,
    λ_instab=λ_instab,
    ε_instab=ε_instab,
    instab_samples=instab_samples,
    epsilons=isnothing(epsilons) ? SemioticTransformer.T[1f-4, 5f-4, 1f-3, 5f-3] : epsilons,
    save_profile=save_profile,
    save_coupling=save_coupling,
    save_trace=save_trace,
    profile_width=profile_width,
)

last = isempty(trace.trace) ? nothing : trace.trace[end]
println("trace length=$(length(trace.trace))")
if last !== nothing
    println("last step=$(last.step) L_total=$(last.L_total) L_time=$(last.L_time) m=$(last.m) s=$(last.s) instab=$(last.instab)")
end
println("profile path=$(trace.profile_path)")
println("coupling path=$(trace.coupling_path)")
if !isempty(trace.table)
    println("trace table preview:\n" * join(split(trace.table, "\n")[1:min(3, end)], "\n"))
end
if save_trace !== nothing
    println("trace saved to $(save_trace)")
end
if save_profile !== nothing
    println("instability profile saved to $(save_profile)")
end
if trace.coupling_path isa AbstractString
    println("coupling matrix saved to $(trace.coupling_path)")
elseif !isempty(coupling_path)
    println("coupling matrix target requested: $(coupling_path)")
end
println("last coupling spark: $(trace.coupling.spark)")
println("last coupling heatmap:\n$(trace.coupling.heatmap)")
