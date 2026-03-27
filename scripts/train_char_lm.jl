#!/usr/bin/env julia
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Random
using Serialization: serialize
using NNlib
using Flux
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

function envbool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    raw in ("1", "true", "t", "yes", "y", "on") && return true
    raw in ("0", "false", "f", "no", "n", "off") && return false
    return default
end

function usage()
    println("""
Usage:
  julia scripts/train_char_lm.jl <text_path_or_dir> [more_paths...]

Env knobs (optional):
  SEMIOTIC_MODEL=semiotic|archetypal   (default: semiotic)
  SEMIOTIC_SEED=2025
  SEMIOTIC_D=64
  SEMIOTIC_LAYERS=2                   (semiotic only)
  SEMIOTIC_H=3                        (semiotic only)
  SEMIOTIC_K=8                        (semiotic only; MeaningField prototypes)
  SEMIOTIC_Z=32                       (semiotic only)
  SEMIOTIC_AR_K=6                     (archetypal only)
  SEMIOTIC_AR_DS=32                   (archetypal only)
  SEMIOTIC_AR_R=32                    (archetypal only)
  SEMIOTIC_STEPS=200
  SEMIOTIC_SEQ=128                    (context length; samples use seq+1)
  SEMIOTIC_BATCH=4
  SEMIOTIC_LR=1e-3
  SEMIOTIC_LOG_EVERY=20
  SEMIOTIC_VAL_FRAC=0.1
  SEMIOTIC_MAX_CHARS=200000
  SEMIOTIC_WILL=true|false
  SEMIOTIC_UPDATE_FIELD=true|false    (semiotic) / SEMIOTIC_UPDATE_FIELDS (archetypal)
  SEMIOTIC_WILL_START=20
  SEMIOTIC_FIELD_START=40
  SEMIOTIC_SAVE=path/to/model.ser     (serialize (model, chars))
""")
end

function collect_text_files(inputs::Vector{String})
    files = String[]
    for input in inputs
        if isdir(input)
            for (root, _, names) in walkdir(input)
                for name in sort(names)
                    endswith(lowercase(name), ".txt") || continue
                    push!(files, joinpath(root, name))
                end
            end
        elseif isfile(input)
            push!(files, input)
        else
            error("path not found: $input")
        end
    end
    isempty(files) && error("no .txt files found in inputs")
    return sort!(unique(files))
end

function load_text(paths::Vector{String}; max_chars::Int=200_000)
    io = IOBuffer()
    written = 0
    for (index, path) in enumerate(paths)
        txt = read(path, String)
        if max_chars > 0
            remaining = max_chars - written
            remaining <= 0 && break
            ncodeunits(txt) > remaining && (txt = first(txt, remaining))
        end
        print(io, txt)
        written += ncodeunits(txt)
        if index < length(paths) && (max_chars <= 0 || written + 2 <= max_chars)
            print(io, "\n\n")
            written += 2
        end
    end
    txt = String(take!(io))
    return txt
end

function build_vocab(txt::AbstractString)
    chars = sort!(collect(Set(txt)))
    char_to_id = Dict{Char, Int}(c => i for (i, c) in enumerate(chars))
    ids = Vector{Int}(undef, length(txt))
    i = 1
    for c in txt
        ids[i] = char_to_id[c]
        i += 1
    end
    return ids, chars
end

function sample_batch(tokens::Vector{Int}, sample_len::Int, batch::Int)
    @assert sample_len >= 2
    max_start = length(tokens) - sample_len + 1
    max_start >= 1 || error("text too short for sample_len=$sample_len (len=$(length(tokens)))")
    seqs = Matrix{Int}(undef, sample_len, batch)
    @inbounds for b in 1:batch
        start = rand(1:max_start)
        seqs[:, b] = @view tokens[start:(start + sample_len - 1)]
    end
    return seqs
end

function maybe_save(path::AbstractString, model, chars)
    isempty(path) && return nothing
    open(path, "w") do io
        serialize(io, (; model, chars))
    end
    return path
end

function train_semiotic(train_tokens, val_tokens, chars; seed::Int, d::Int, layers::Int, H::Int, k::Int, z::Int,
        steps::Int, seq::Int, batch::Int, lr::Float64, log_every::Int, will::Bool, update_field::Bool,
        will_start::Int, field_start::Int, save_path::AbstractString)
    Random.seed!(seed)

    vocab = length(chars)
    square = vocab >= 4 ? SemioticTransformer.SemioticSquare(1, 2, 3, 4) : SemioticTransformer.SemioticSquare(1, 1, 1, 1)
    model = SemioticTransformer.SemioticModel(vocab, d; layers=layers, H=H, k=k, z=z, classes=vocab, square=square)
    rule = Flux.Optimisers.OptimiserChain(Flux.Optimisers.ClipNorm(1.0), Flux.Optimisers.Adam(lr))
    opt = Flux.setup(rule, model)

    sample_len = seq + 1
    for step in 1:steps
        use_will = will && step >= will_start
        use_field = update_field && step >= field_start
        batch_tokens = sample_batch(train_tokens, sample_len, batch)
        grads = Flux.gradient(model) do m
            L, _ = SemioticTransformer.lossfn(m, batch_tokens; update_field=false, will=use_will)
            L
        end
        if SemioticTransformer._tree_all_finite(grads[1])
            Flux.update!(opt, model, grads[1])
        end
        SemioticTransformer.forward(model, batch_tokens; update_field=use_field, will=use_will)

        if log_every > 0 && step % log_every == 0
            train_L, train_parts = SemioticTransformer.lossfn(model, batch_tokens; update_field=false, will=use_will)
            val_batch = sample_batch(val_tokens, sample_len, batch)
            val_L, val_parts = SemioticTransformer.lossfn(model, val_batch; update_field=false, will=false)
            @info "step=$step" train_loss=train_L train_Lce=train_parts.Lce val_loss=val_L val_Lce=val_parts.Lce will=use_will update_field=use_field
        end
    end

    saved = maybe_save(save_path, model, chars)
    !isnothing(saved) && @info "saved" path=saved
    return model
end

function train_archetypal(train_tokens, val_tokens, chars; seed::Int, d::Int, K::Int, ds::Int, r::Int,
        steps::Int, seq::Int, batch::Int, lr::Float64, log_every::Int, will::Bool, update_fields::Bool,
        will_start::Int, field_start::Int, save_path::AbstractString)
    Random.seed!(seed)

    vocab = length(chars)
    model = SemioticTransformer.Archetypal.ArchetypalModel(vocab, d; K=K, ds=ds, r=r)
    rule = Flux.Optimisers.OptimiserChain(Flux.Optimisers.ClipNorm(1.0), Flux.Optimisers.Adam(lr))
    opt = Flux.setup(rule, model)

    sample_len = seq + 1
    for step in 1:steps
        use_will = will && step >= will_start
        use_fields = update_fields && step >= field_start
        batch_tokens = sample_batch(train_tokens, sample_len, batch)
        grads = Flux.gradient(model) do m
            L, _ = SemioticTransformer.Archetypal.lossfn(m, batch_tokens; update_fields=false, will=use_will)
            L
        end
        if SemioticTransformer._tree_all_finite(grads[1])
            Flux.update!(opt, model, grads[1])
        end
        SemioticTransformer.Archetypal.forward(model, batch_tokens; update_fields=use_fields, will=use_will)

        if log_every > 0 && step % log_every == 0
            train_L, train_parts = SemioticTransformer.Archetypal.lossfn(model, batch_tokens; update_fields=false, will=use_will)
            val_batch = sample_batch(val_tokens, sample_len, batch)
            val_L, val_parts = SemioticTransformer.Archetypal.lossfn(model, val_batch; update_fields=false, will=false)
            @info "step=$step" train_loss=train_L train_Lce=train_parts.Lce val_loss=val_L val_Lce=val_parts.Lce will=use_will update_fields=use_fields
        end
    end

    saved = maybe_save(save_path, model, chars)
    !isnothing(saved) && @info "saved" path=saved
    return model
end

function main()
    if isempty(ARGS)
        usage()
        error("missing <text_path_or_dir>")
    end
    inputs = collect(String, ARGS)
    model_kind = lowercase(strip(envstr("SEMIOTIC_MODEL", "semiotic")))
    seed = envint("SEMIOTIC_SEED", 2025)
    d = envint("SEMIOTIC_D", 64)
    steps = envint("SEMIOTIC_STEPS", 200)
    seq = envint("SEMIOTIC_SEQ", 128)
    batch = envint("SEMIOTIC_BATCH", 4)
    lr = envfloat("SEMIOTIC_LR", 1e-3)
    log_every = envint("SEMIOTIC_LOG_EVERY", 20)
    val_frac = envfloat("SEMIOTIC_VAL_FRAC", 0.1)
    max_chars = envint("SEMIOTIC_MAX_CHARS", 200_000)
    will = envbool("SEMIOTIC_WILL", true)
    will_start = envint("SEMIOTIC_WILL_START", 20)
    field_start = envint("SEMIOTIC_FIELD_START", 40)
    save_path = envstr("SEMIOTIC_SAVE", "")

    text_files = collect_text_files(inputs)
    txt = load_text(text_files; max_chars=max_chars)
    tokens, chars = build_vocab(txt)
    vocab = length(chars)
    vocab >= 2 || error("vocab too small (need >=2 unique chars)")

    val_n = max(10 * (seq + 1), round(Int, length(tokens) * val_frac))
    val_n = min(val_n, length(tokens) ÷ 2)
    train_n = length(tokens) - val_n
    train_n > seq + 2 || error("not enough training data (len=$(length(tokens))) for seq=$seq")

    train_tokens = tokens[1:train_n]
    val_tokens = tokens[(train_n + 1):end]

    @info "dataset" inputs=inputs files=length(text_files) max_chars=max_chars n_tokens=length(tokens) vocab=vocab train=train_n val=val_n

    if model_kind == "semiotic"
        layers = envint("SEMIOTIC_LAYERS", 2)
        H = envint("SEMIOTIC_H", 3)
        k = envint("SEMIOTIC_K", 8)
        z = envint("SEMIOTIC_Z", 32)
        update_field = envbool("SEMIOTIC_UPDATE_FIELD", true)
        @info "model" kind=model_kind d=d layers=layers H=H k=k z=z steps=steps seq=seq batch=batch lr=lr
        train_semiotic(train_tokens, val_tokens, chars;
            seed=seed, d=d, layers=layers, H=H, k=k, z=z,
            steps=steps, seq=seq, batch=batch, lr=lr, log_every=log_every, will=will,
            update_field=update_field, will_start=will_start, field_start=field_start, save_path=save_path,
        )
    elseif model_kind == "archetypal"
        K = envint("SEMIOTIC_AR_K", 6)
        ds = envint("SEMIOTIC_AR_DS", d ÷ 2)
        r = envint("SEMIOTIC_AR_R", 32)
        update_fields = envbool("SEMIOTIC_UPDATE_FIELDS", true)
        @info "model" kind=model_kind d=d K=K ds=ds r=r steps=steps seq=seq batch=batch lr=lr
        train_archetypal(train_tokens, val_tokens, chars;
            seed=seed, d=d, K=K, ds=ds, r=r,
            steps=steps, seq=seq, batch=batch, lr=lr, log_every=log_every, will=will,
            update_fields=update_fields, will_start=will_start, field_start=field_start, save_path=save_path,
        )
    else
        usage()
        error("unknown SEMIOTIC_MODEL=$model_kind (expected semiotic|archetypal)")
    end
end

main()
