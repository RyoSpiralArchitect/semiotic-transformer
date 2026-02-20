#!/usr/bin/env julia
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Random
using Serialization: deserialize
using NNlib
using SemioticTransformer

function envint(name::AbstractString, default::Integer)
    raw = get(ENV, name, string(default))
    v = try
        parse(Int, raw)
    catch
        Int(default)
    end
    return v
end

function envfloat(name::AbstractString, default::Real)
    raw = get(ENV, name, string(default))
    v = try
        parse(Float64, raw)
    catch
        Float64(default)
    end
    return v
end

function envstr(name::AbstractString, default::AbstractString)
    return get(ENV, name, default)
end

function usage()
    println("""
Usage:
  julia scripts/generate_char_lm.jl <model.ser> [prompt...]

Env knobs (optional):
  SEMIOTIC_PROMPT="hello"
  SEMIOTIC_GEN_LEN=200
  SEMIOTIC_SEQ=128
  SEMIOTIC_TEMP=1.0
  SEMIOTIC_SEED=0
""")
end

function sample_categorical(probs::AbstractVector{<:Real})
    s = 0.0
    r = rand()
    @inbounds for i in eachindex(probs)
        s += probs[i]
        if r <= s
            return Int(i)
        end
    end
    return Int(lastindex(probs))
end

function encode_prompt(prompt::AbstractString, char_to_id::Dict{Char, Int})
    ids = Int[]
    bad = Set{Char}()
    for c in prompt
        id = get(char_to_id, c, 0)
        if id == 0
            push!(bad, c)
        else
            push!(ids, id)
        end
    end
    isempty(bad) || error("prompt contains chars not in vocab: $(collect(bad))")
    return ids
end

function decode(ids::Vector{Int}, chars::Vector{Char})
    io = IOBuffer()
    for id in ids
        1 <= id <= length(chars) || continue
        print(io, chars[id])
    end
    return String(take!(io))
end

function next_token(model, ctx::Vector{Int}; temp::Float64)
    if model isa SemioticTransformer.SemioticModel
        logits, _, _, _ = SemioticTransformer.forward(model, ctx; update_field=false, will=false)
        p = NNlib.softmax(view(logits, :, size(logits, 2)) ./ temp)
        return sample_categorical(p)
    elseif model isa SemioticTransformer.Archetypal.ArchetypalModel
        logits, _, _, _ = SemioticTransformer.Archetypal.forward(model, ctx; update_fields=false, will=false)
        p = NNlib.softmax(view(logits, :, size(logits, 2)) ./ temp)
        return sample_categorical(p)
    else
        error("unsupported model type: $(typeof(model))")
    end
end

function main()
    if isempty(ARGS)
        usage()
        error("missing <model.ser>")
    end
    model_path = ARGS[1]
    prompt = isempty(ARGS[2:end]) ? envstr("SEMIOTIC_PROMPT", "") : join(ARGS[2:end], " ")
    isempty(prompt) && (usage(); error("missing prompt (args or SEMIOTIC_PROMPT)"))

    seed = envint("SEMIOTIC_SEED", 0)
    seed != 0 && Random.seed!(seed)
    gen_len = envint("SEMIOTIC_GEN_LEN", 200)
    ctx_len = envint("SEMIOTIC_SEQ", 128)
    temp = envfloat("SEMIOTIC_TEMP", 1.0)
    temp > 0 || error("SEMIOTIC_TEMP must be > 0")

    bundle = open(model_path, "r") do io
        deserialize(io)
    end
    model = bundle.model
    chars = bundle.chars
    char_to_id = Dict{Char, Int}(c => i for (i, c) in enumerate(chars))

    ids = encode_prompt(prompt, char_to_id)
    for _ in 1:gen_len
        ctx = length(ids) > ctx_len ? ids[(end - ctx_len + 1):end] : ids
        push!(ids, next_token(model, ctx; temp=temp))
    end
    println(decode(ids, chars))
end

main()
