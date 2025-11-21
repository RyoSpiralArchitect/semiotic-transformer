module SemioticTransformer

using Flux
using LinearAlgebra
using Statistics
using Random
using NNlib # for softplus, softmax
using Functors
const T = Float32

# -----------------------------
# Negation（involution-ish）
# -----------------------------
struct Negation
    N::Matrix{T}
end
@functor Negation
Negation(d::Int; init=Flux.glorot_uniform) = Negation(init(d, d))
negate(neg::Negation, x::AbstractVecOrMat{T}) = neg.N * x

function negation_penalty(neg::Negation; λ_inv=T(1e-3), λ_iso=T(1e-3), λ_sym=T(1e-4))
    d = size(neg.N, 1)
    I = Matrix{T}(I, d, d)
    L_inv = norm(neg.N * neg.N .- I)^2
    L_iso = norm(neg.N' * neg.N .- I)^2
    L_sym = norm(neg.N .- neg.N')^2
    return λ_inv * L_inv + λ_iso * L_iso + λ_sym * L_sym
end

# -----------------------------
# Difference field（差異場）
# -----------------------------
struct DifferenceField
    M::Matrix{T}
    φ::Function
end
@functor DifferenceField
DifferenceField(d::Int; φ = NNlib.softplus) = DifferenceField(Matrix{T}(I, d, d), φ)

function difference_matrix(df::DifferenceField, X::AbstractMatrix{T})
    _, n = size(X)
    M = df.M
    D = Matrix{T}(undef, n, n)
    @inbounds @views for i in 1:n
        xi = X[:, i]
        D[i, i] = df.φ(zero(T))
        i == n && continue
        for j in (i + 1):n
            Δ = xi .- X[:, j]
            val = df.φ(dot(Δ, M * Δ))
            D[i, j] = val
            D[j, i] = val
        end
    end
    return D
end

function difference_matrix(df::DifferenceField, X::AbstractArray{T,3})
    n = size(X, 2)
    batches = size(X, 3)
    D = Array{T}(undef, n, n, batches)
    @inbounds @views for b in 1:batches
        D[:, :, b] = difference_matrix(df, view(X, :, :, b))
    end
    return D
end

# -----------------------------
# Meaning field（意味ポテンシャル）
# -----------------------------
struct MeaningField
    P::Matrix{T}   # d×k
    w::Vector{T}   # k
    scale::T
end
@functor MeaningField
MeaningField(d::Int, k::Int; scale=T(1.0)) =
    MeaningField(Flux.glorot_uniform(d, k), ones(T, k), scale)

function potential(mf::MeaningField, X::AbstractMatrix{T})
    d, n = size(X)
    k = size(mf.P, 2)
    Φ = zeros(T, n)
    @inbounds for j in 1:n
        x = view(X, :, j)
        s = zero(T)
        for t in 1:k
            p = view(mf.P, :, t)
            δ = x .- p
            s += -mf.w[t] * (δ' * δ)
        end
        Φ[j] = mf.scale * s
    end
    return Φ
end

function potential(mf::MeaningField, X::AbstractArray{T,3})
    n = size(X, 2)
    batches = size(X, 3)
    Φ = zeros(T, n, batches)
    @inbounds @views for b in 1:batches
        Φ[:, b] = potential(mf, view(X, :, :, b))
    end
    return Φ
end

function potential_grad(mf::MeaningField, x::AbstractVector{T})
    g = zeros(T, length(x))
    @inbounds for t in 1:size(mf.P, 2)
        g .+= -2f0 * mf.w[t] .* (x .- view(mf.P, :, t))
    end
    return mf.scale .* g
end

function meaning_instability(mf::MeaningField, X::AbstractMatrix{T}; ε::T=T(1e-3), samples::Int=1)
    d, n = size(X)
    accum = zero(T)
    @inbounds for _ in 1:samples
        noise = ε .* randn(T, d, n)
        for j in 1:n
            x = view(X, :, j)
            Δ = potential_grad(mf, x) .- potential_grad(mf, x .+ view(noise, :, j))
            accum += sum(abs2, Δ)
        end
    end
    return accum / (samples * n)
end

function meaning_instability(mf::MeaningField, X::AbstractArray{T,3}; ε::T=T(1e-3), samples::Int=1)
    d, n, batches = size(X)
    accum = zero(T)
    @inbounds for _ in 1:samples
        noise = ε .* randn(T, d, n, batches)
        for b in 1:batches, j in 1:n
            x = view(X, :, j, b)
            Δ = potential_grad(mf, x) .- potential_grad(mf, x .+ view(noise, :, j, b))
            accum += sum(abs2, Δ)
        end
    end
    return accum / (samples * n * batches)
end

function update!(mf::MeaningField, X::AbstractMatrix{T}; τ=T(0.95), temp=T(1.0))
    d, n = size(X)
    k = size(mf.P, 2)
    R = zeros(T, k, n)
    for j in 1:n
        x = view(X, :, j)
        for t in 1:k
            δ = x .- view(mf.P, :, t)
            R[t, j] = -sum(abs2, δ) / temp
        end
    end
    R .= NNlib.softmax(R; dims=1)
    Nk = sum(R; dims=2)
    newP = similar(mf.P)
    for t in 1:k
        if Nk[t] > eps(T)
            acc = zeros(T, d)
            for j in 1:n
                acc .+= R[t, j] .* view(X, :, j)
            end
            newP[:, t] = acc ./ Nk[t]
        else
            newP[:, t] = mf.P[:, t]
        end
    end
    mf.P .= τ .* mf.P .+ (one(T) - τ) .* newP
    return nothing
end

function update!(mf::MeaningField, X::AbstractArray{T,3}; τ=T(0.95), temp=T(1.0))
    d, n, batches = size(X)
    k = size(mf.P, 2)
    R = zeros(T, k, n, batches)
    for b in 1:batches
        for j in 1:n
            x = view(X, :, j, b)
            for t in 1:k
                δ = x .- view(mf.P, :, t)
                R[t, j, b] = -sum(abs2, δ) / temp
            end
        end
    end
    R .= NNlib.softmax(R; dims=1)
    Nk = sum(R; dims=(2, 3))
    newP = similar(mf.P)
    for t in 1:k
        if Nk[t] > eps(T)
            acc = zeros(T, d)
            for b in 1:batches, j in 1:n
                acc .+= R[t, j, b] .* view(X, :, j, b)
            end
            newP[:, t] = acc ./ Nk[t]
        else
            newP[:, t] = mf.P[:, t]
        end
    end
    mf.P .= τ .* mf.P .+ (one(T) - τ) .* newP
    return nothing
end

# -----------------------------
# Semiotic square（Greimas）
# -----------------------------
struct SemioticSquare
    s1::Int
    s2::Int
    ns1::Int
    ns2::Int
end

function square_loss(X::AbstractMatrix{T}, sq::SemioticSquare; margin_contra=T(2.0), margin_imp=T(0.5))
    A  = view(X, :, sq.s1); nA = view(X, :, sq.ns1)
    B  = view(X, :, sq.s2); nB = view(X, :, sq.ns2)
    d(x, y) = norm(x .- y)

    L_contra = NNlib.relu(margin_contra .- d(A, nA))^2 + NNlib.relu(margin_contra .- d(B, nB))^2
    L_contrary = 0.5f0 * (NNlib.relu(margin_contra .- d(A, B))^2 + NNlib.relu(margin_contra .- d(nA, nB))^2)
    L_impl = (NNlib.relu(d(A, nB) .- margin_imp)^2 + NNlib.relu(d(B, nA) .- margin_imp)^2)
    return L_contra + L_contrary + L_impl
end

function square_penalty(X::AbstractMatrix{T}, squares::Vector{SemioticSquare})
    isempty(squares) && return zero(T)
    loss = zero(T)
    @inbounds for sq in squares
        loss += square_loss(X, sq)
    end
    return loss
end

# -----------------------------
# A) Kant: Noumenon（VAE-ish latent）
# -----------------------------
struct Noumenon
    Ezμ::Dense
    Ezlogσ::Dense
    Dz::Dense
end
@functor Noumenon
function Noumenon(d::Int, z::Int)
    Noumenon(Dense(d, z), Dense(d, z), Dense(z, d))
end

sample_z(μ::AbstractVecOrMat{T}, logσ::AbstractVecOrMat{T}) = μ .+ exp.(0.5f0 .* logσ) .* randn(T, size(μ))

function encode_decode(nm::Noumenon, X::AbstractMatrix{T})
    μ = nm.Ezμ(X)
    logσ = nm.Ezlogσ(X)
    Z = sample_z(μ, logσ)
    Xhat = nm.Dz(Z)
    KL = 0.5f0 * sum(exp.(logσ) .+ μ.^2 .- one(T) .- logσ) / size(X, 2)
    return Z, Xhat, KL
end

function encode_decode(nm::Noumenon, X::AbstractArray{T,3})
    d, n, batches = size(X)
    cols = n * batches
    Xflat = reshape(X, d, cols)
    μ = nm.Ezμ(Xflat)
    logσ = nm.Ezlogσ(Xflat)
    Z = sample_z(μ, logσ)
    Xhat = nm.Dz(Z)
    KL = 0.5f0 * sum(exp.(logσ) .+ μ.^2 .- one(T) .- logσ) / cols
    return reshape(Z, size(μ, 1), n, batches), reshape(Xhat, d, n, batches), KL
end

# -----------------------------
# B) Jung: Self & coniunctio
# -----------------------------
struct SelfField
    s::Vector{T}
    α::T
end
@functor SelfField
SelfField(d::Int; α=T(0.5)) = SelfField(Flux.glorot_uniform(d), α)

function coniunctio(neg::Negation, x::AbstractVector{T}; α::T=T(0.5))
    y = (one(T) - α) .* x .+ α .* (neg.N * x)
    return y ./ max(norm(y), eps(T))
end

function self_loss(sf::SelfField, neg::Negation, Φgrad::Function, A::AbstractVector{T}, B::AbstractVector{T}; λ_eq=T(1.0), λ_stat=T(0.1))
    cA = coniunctio(neg, A; α=sf.α)
    cB = coniunctio(neg, B; α=sf.α)
    L_eq = norm(sf.s .- cA)^2 + norm(sf.s .- cB)^2
    L_stat = norm(Φgrad(sf.s))^2
    return λ_eq * L_eq + λ_stat * L_stat
end

# -----------------------------
# C) Schopenhauer: Will-flow
# -----------------------------
function diff_grad(df::DifferenceField, X::AbstractMatrix{T})
    M = df.M
    d, n = size(X)
    G = zeros(T, d, n)
    μ = mean(X; dims=2)
    @inbounds for j in 1:n
        G[:, j] .= 2f0 .* (M * (n .* view(X, :, j) .- vec(μ)))
    end
    return G
end

function diff_grad(df::DifferenceField, X::AbstractArray{T,3})
    d, n, batches = size(X)
    G = zeros(T, d, n, batches)
    @inbounds for b in 1:batches
        G[:, :, b] .= diff_grad(df, view(X, :, :, b))
    end
    return G
end

function will_step!(X::AbstractMatrix{T}, mf::MeaningField, df::DifferenceField; η=T(1e-2), ρ=T(1e-3))
    d, n = size(X)
    Gφ = similar(X)
    for j in 1:n
        Gφ[:, j] = potential_grad(mf, view(X, :, j))
    end
    GD = diff_grad(df, X)
    X .+= η .* Gφ .- ρ .* GD
    return X
end

function will_step!(X::AbstractArray{T,3}, mf::MeaningField, df::DifferenceField; η=T(1e-2), ρ=T(1e-3))
    d, n, batches = size(X)
    for b in 1:batches
        tmp = view(X, :, :, b)
        will_step!(tmp, mf, df; η=η, ρ=ρ)
    end
    return X
end

# -----------------------------
# D) Fechner/Weber: JND constraint
# -----------------------------
function jnd_loss(df::DifferenceField, X::AbstractMatrix{T}; k=T(0.1), θ=T(0.05))
    d, n = size(X)
    L = zero(T)
    for j in 1:n
        x = view(X, :, j)
        δ = k * max(norm(x), T(1.0)) .* randn(T, d) ./ sqrt(T(d))
        xp = x .+ δ
        Dloc = dot((x - xp), df.M * (x - xp)) |> df.φ
        L += (Dloc - θ)^2
    end
    return L / n
end

function jnd_loss(df::DifferenceField, X::AbstractArray{T,3}; k=T(0.1), θ=T(0.05))
    batches = size(X, 3)
    L = zero(T)
    for b in 1:batches
        L += jnd_loss(df, view(X, :, :, b); k=k, θ=θ)
    end
    return L / batches
end

# -----------------------------
# Semiotic heads / attention with roles
# -----------------------------
@enum HeadRole::UInt8 begin Contradiction=1; Contrary=2; Implication=3; end

struct SemioticHead
    WQ::Matrix{T}
    WK::Matrix{T}
    WV::Matrix{T}
    df::DifferenceField
    β::T
    γ::T
    η::T
    role::HeadRole
end
@functor SemioticHead
function SemioticHead(d::Int, dk::Int, dv::Int, role::HeadRole; β=T(0.2), γ=T(0.2), η=T(0.2))
    SemioticHead(Flux.glorot_uniform(dk, d), Flux.glorot_uniform(dk, d), Flux.glorot_uniform(dv, d), DifferenceField(dk), β, γ, η, role)
end

struct SemioticMHA
    heads::Vector{SemioticHead}
    WO::Matrix{T}
    neg::Negation
end
@functor SemioticMHA
function SemioticMHA(d::Int; H::Int=3, dk::Int=div(d, H), dv::Int=div(d, H))
    roles = [Contradiction, Contrary, Implication][1:H]
    hs = [SemioticHead(d, dk, dv, r;
                       β = r == Contradiction ? T(0.8) : r == Contrary ? T(0.4) : T(0.2),
                       γ = r == Implication   ? T(0.6) : T(0.2),
                       η = T(0.3)) for r in roles]
    SemioticMHA(hs, Flux.glorot_uniform(d, dv * H), Negation(d))
end

function role_bias(h::SemioticHead, Q::AbstractMatrix{T}, K::AbstractMatrix{T}, Φ::AbstractVector{T}, neg::Negation)
    n = size(Q, 2)
    Φrow = reshape(Φ, 1, :)
    Φpair = (Φrow .+ Φrow') .* 0.5f0
    S = (Q' * K) ./ sqrt(T(size(Q, 1)))
    D = difference_matrix(h.df, K)
    R = zeros(T, n, n)
    if h.role == Implication
        NK = negate(neg, K)
        S_imp = (Q' * NK) ./ sqrt(T(size(Q, 1)))
        R .= h.η .* S_imp
    elseif h.role == Contradiction
        R .= -h.η .* D
    elseif h.role == Contrary
        R .= -0.5f0 .* h.η .* D
    end
    return S .- h.β .* D .+ h.γ .* Φpair .+ R
end

function (mha::SemioticMHA)(X::AbstractMatrix{T}, Φ::AbstractVector{T})
    parts = Matrix{T}[]
    for h in mha.heads
        Q, K, V = h.WQ * X, h.WK * X, h.WV * X
        A = role_bias(h, Q, K, Φ, mha.neg)
        P = NNlib.softmax(A; dims=2)
        push!(parts, V * P')
    end
    return mha.WO * reduce(vcat, parts)
end

function (mha::SemioticMHA)(X::AbstractArray{T,3}, Φ::AbstractMatrix{T})
    d, n, batches = size(X)
    cols = n * batches
    dk = size(mha.heads[1].WQ, 1)
    dv = size(mha.heads[1].WV, 1)
    Xflat = reshape(X, d, cols)
    Qs = [reshape(h.WQ * Xflat, dk, n, batches) for h in mha.heads]
    Ks = [reshape(h.WK * Xflat, dk, n, batches) for h in mha.heads]
    Vs = [reshape(h.WV * Xflat, dv, n, batches) for h in mha.heads]
    out = zeros(T, d, n, batches)
    @inbounds for b in 1:batches
        Φcol = view(Φ, :, b)
        Φrow = reshape(Φcol, 1, :)
        for idx in 1:length(mha.heads)
            h = mha.heads[idx]
            Q = Qs[idx][:, :, b]
            K = Ks[idx][:, :, b]
            V = Vs[idx][:, :, b]
            A = role_bias(h, Q, K, Φcol, mha.neg)
            P = NNlib.softmax(A; dims=2)
            out[:, :, b] .+= mha.WO[:, ((idx - 1) * dv + 1):(idx * dv)] * (V * P')
        end
    end
    return out
end

# -----------------------------
# Meaning chain
# -----------------------------
struct MeaningChainLayer
    den::Dense
    con::Dense
    myth::Dense
    α::Vector{T}
end
@functor MeaningChainLayer
function MeaningChainLayer(d::Int; h::Int=d)
    MeaningChainLayer(Dense(d, d), Dense(d, h, gelu), Dense(h, d), param([T(0.6), T(0.3), T(0.1)]))
end

function (m::MeaningChainLayer)(X::AbstractMatrix{T}, Φ::AbstractVector{T})
    Φrow = reshape(Φ, 1, :)
    d0 = m.den(X)
    c0 = m.con(X) .* (1 .+ Φrow)
    y0 = m.myth(c0)
    α = NNlib.softmax(m.α)
    return α[1] .* d0 .+ α[2] .* c0 .+ α[3] .* y0
end

function (m::MeaningChainLayer)(X::AbstractArray{T,3}, Φ::AbstractMatrix{T})
    d, n, batches = size(X)
    cols = n * batches
    Xflat = reshape(X, d, cols)
    Φrow = reshape(vec(Φ), 1, cols)
    d0 = m.den(Xflat)
    c0 = m.con(Xflat) .* (1 .+ Φrow)
    y0 = m.myth(c0)
    α = NNlib.softmax(m.α)
    out = α[1] .* d0 .+ α[2] .* c0 .+ α[3] .* y0
    return reshape(out, d, n, batches)
end

# -----------------------------
# Transformer block（Semiotic）
# -----------------------------
struct SemioticBlock
    nm::Noumenon
    mha::SemioticMHA
    norm1::LayerNorm
    chain::MeaningChainLayer
    norm2::LayerNorm
    mf::MeaningField
    df::DifferenceField
    sf::SelfField
end
@functor SemioticBlock
function SemioticBlock(d::Int; z::Int=div(d, 2), H::Int=3, dk::Int=div(d, H), dv::Int=div(d, H), h::Int=d, k::Int=8)
    SemioticBlock(Noumenon(d, z), SemioticMHA(d; H=H, dk=dk, dv=dv), LayerNorm(d), MeaningChainLayer(d; h=h), LayerNorm(d), MeaningField(d, k), DifferenceField(d), SelfField(d))
end

_apply_layernorm(norm::LayerNorm, X::AbstractMatrix{T}) where {T} = norm(X)
function _apply_layernorm(norm::LayerNorm, X::AbstractArray{T,3}) where {T}
    d, n, batches = size(X)
    cols = n * batches
    return reshape(norm(reshape(X, d, cols)), d, n, batches)
end

function (b::SemioticBlock)(X::AbstractMatrix{T}; update_field::Bool=false, will::Bool=false)
    Z, Xhat, KL = encode_decode(b.nm, X)
    Φ = potential(b.mf, X)
    Y = X .+ b.mha(_apply_layernorm(b.norm1, X), Φ)
    Z2 = Y .+ b.chain(_apply_layernorm(b.norm2, Y), Φ)
    if will
        will_step!(Z2, b.mf, b.df)
    end
    if update_field
        update!(b.mf, Z2)
    end
    recL = mean((X .- Xhat).^2)
    return Z2, KL, recL
end

function (b::SemioticBlock)(X::AbstractArray{T,3}; update_field::Bool=false, will::Bool=false)
    Z, Xhat, KL = encode_decode(b.nm, X)
    Φ = potential(b.mf, X)
    Y = X .+ b.mha(_apply_layernorm(b.norm1, X), Φ)
    Z2 = Y .+ b.chain(_apply_layernorm(b.norm2, Y), Φ)
    if will
        will_step!(Z2, b.mf, b.df)
    end
    if update_field
        update!(b.mf, Z2)
    end
    recL = mean((X .- Xhat).^2)
    return Z2, KL, recL
end

# -----------------------------
# Model
# -----------------------------
struct SemioticModel
    emb::Embedding
    blocks::Vector{SemioticBlock}
    ln::LayerNorm
    proj::Dense
    squares::Vector{SemioticSquare}
end
@functor SemioticModel

function (emb::Embedding)(tokens::AbstractMatrix{<:Integer})
    seq_len, batches = size(tokens)
    d = size(emb.weight, 1)
    X = emb(vec(tokens))
    return reshape(X, d, seq_len, batches)
end

function SemioticModel(vocab::Int, d::Int; layers::Int=2, H::Int=3, k::Int=8, z::Int=div(d, 2), classes::Int=vocab, square=nothing, squares=nothing)
    blocks = [SemioticBlock(d; z=z, H=H, k=k) for _ in 1:layers]
    sqs = SemioticSquare[]
    if squares !== nothing
        append!(sqs, squares)
    elseif square !== nothing
        push!(sqs, square)
    end
    SemioticModel(Flux.Embedding(vocab, d), blocks, LayerNorm(d), Dense(d, classes), sqs)
end

function forward(m::SemioticModel, tokens::AbstractVector{<:Integer}; update_field=false, will=false)
    X = m.emb(tokens)
    total_KL = zero(T)
    recL = zero(T)
    for b in m.blocks
        X, KL, rL = b(X; update_field=update_field, will=will)
        total_KL += KL
        recL += rL
    end
    X = _apply_layernorm(m.ln, X)
    logits = m.proj(X)
    return logits, total_KL, recL, X
end

function forward(m::SemioticModel, tokens::AbstractMatrix{<:Integer}; update_field=false, will=false)
    X = m.emb(tokens)
    total_KL = zero(T)
    recL = zero(T)
    for b in m.blocks
        X, KL, rL = b(X; update_field=update_field, will=will)
        total_KL += KL
        recL += rL
    end
    X = _apply_layernorm(m.ln, X)
    logits = _apply_dense(m.proj, X)
    return logits, total_KL, recL, X
end

_apply_dense(layer::Dense, X::AbstractMatrix{T}) where {T} = layer(X)
function _apply_dense(layer::Dense, X::AbstractArray{T,3}) where {T}
    d, n, batches = size(X)
    cols = n * batches
    return reshape(layer(reshape(X, d, cols)), size(layer.weight, 1), n, batches)
end

# -----------------------------
# Loss helpers
# -----------------------------
function next_token_pairs(tokens::AbstractVector{<:Integer})
    length(tokens) > 1 || error("Need at least two tokens to form training pairs")
    return tokens[1:end-1], tokens[2:end]
end

function next_token_pairs(tokens::AbstractMatrix{<:Integer})
    size(tokens, 1) > 1 || error("Need at least two tokens to form training pairs")
    return tokens[1:end-1, :], tokens[2:end, :]
end

function _ce_loss(logits::AbstractMatrix{T}, targets::AbstractVector{<:Integer}; pad_token::Union{Nothing,Int}=nothing) where {T}
    classes = size(logits, 1)
    if pad_token === nothing
        return Flux.logitcrossentropy(logits, onehotbatch(targets, 1:classes))
    end
    keep = findall(!=(pad_token), targets)
    isempty(keep) && return zero(T)
    return Flux.logitcrossentropy(logits[:, keep], onehotbatch(view(targets, keep), 1:classes))
end

function _ce_loss(logits::AbstractArray{T,3}, targets::AbstractMatrix{<:Integer}; pad_token::Union{Nothing,Int}=nothing) where {T}
    classes = size(logits, 1)
    cols = size(logits, 2) * size(logits, 3)
    flat_logits = reshape(logits, classes, cols)
    flat_targets = vec(targets)
    if pad_token !== nothing
        mask = flat_targets .!= pad_token
        if !any(mask)
            return zero(T)
        end
        flat_logits = view(flat_logits, :, mask)
        flat_targets = view(flat_targets, mask)
    end
    return Flux.logitcrossentropy(flat_logits, onehotbatch(flat_targets, 1:classes))
end

function _self_loss_block(b::SemioticBlock, X::AbstractMatrix{T})
    size(X, 2) < 2 && return zero(T)
    A = view(X, :, 1); B = view(X, :, 2)
    return self_loss(b.sf, b.mha.neg, x -> potential_grad(b.mf, x), A, B; λ_eq=T(1.0), λ_stat=T(0.1))
end

function _self_loss_block(b::SemioticBlock, X::AbstractArray{T,3})
    size(X, 2) < 2 && return zero(T)
    batches = size(X, 3)
    L = zero(T)
    for i in 1:batches
        L += _self_loss_block(b, view(X, :, :, i))
    end
    return L / batches
end

function _negation_penalty(blocks::Vector{SemioticBlock}; kw...)
    L = zero(T)
    for b in blocks
        L += negation_penalty(b.mha.neg; kw...)
    end
    return L
end

function lossfn(m::SemioticModel, tokens::AbstractVector{<:Integer}, targets::AbstractVector{<:Integer}; λ_square=T(0.05), λ_neg=T(1e-3), λ_KL=T(1e-3), λ_rec=T(1e-2), λ_self=T(1e-2), λ_jnd=T(1e-3), λ_instab::T=T(0.0), ε_instab::T=T(1e-3), instab_samples::Int=1, pad_token::Union{Nothing,Int}=nothing, update_field::Bool=false, will::Bool=true)
    logits, KL, recL, X = forward(m, tokens; update_field=update_field, will=will)
    Lce = _ce_loss(logits, targets; pad_token=pad_token)
    top_block = m.blocks[end]
    Lself = _self_loss_block(top_block, X)
    Lneg = _negation_penalty(m.blocks; λ_inv=T(1e-3), λ_iso=T(1e-3), λ_sym=T(1e-4))
    Lsq = square_penalty(m.emb.weight, m.squares)
    Ljnd = jnd_loss(top_block.df, X; k=T(0.08), θ=T(0.05))
    Linstab = iszero(λ_instab) ? zero(T) : meaning_instability(top_block.mf, X; ε=ε_instab, samples=instab_samples)
    L = Lce + λ_KL * KL + λ_rec * recL + λ_square * Lsq + λ_neg * Lneg + λ_self * Lself + λ_jnd * Ljnd + λ_instab * Linstab
    return L, (; Lce, KL, recL, Lsq, Lneg, Lself, Ljnd, Linstab)
end

function lossfn(m::SemioticModel, tokens::AbstractMatrix{<:Integer}, targets::AbstractMatrix{<:Integer}; λ_square=T(0.05), λ_neg=T(1e-3), λ_KL=T(1e-3), λ_rec=T(1e-2), λ_self=T(1e-2), λ_jnd=T(1e-3), λ_instab::T=T(0.0), ε_instab::T=T(1e-3), instab_samples::Int=1, pad_token::Union{Nothing,Int}=nothing, update_field::Bool=false, will::Bool=true)
    logits, KL, recL, X = forward(m, tokens; update_field=update_field, will=will)
    Lce = _ce_loss(logits, targets; pad_token=pad_token)
    top_block = m.blocks[end]
    Lself = _self_loss_block(top_block, X)
    Lneg = _negation_penalty(m.blocks; λ_inv=T(1e-3), λ_iso=T(1e-3), λ_sym=T(1e-4))
    Lsq = square_penalty(m.emb.weight, m.squares)
    Ljnd = jnd_loss(top_block.df, X; k=T(0.08), θ=T(0.05))
    Linstab = iszero(λ_instab) ? zero(T) : meaning_instability(top_block.mf, X; ε=ε_instab, samples=instab_samples)
    L = Lce + λ_KL * KL + λ_rec * recL + λ_square * Lsq + λ_neg * Lneg + λ_self * Lself + λ_jnd * Ljnd + λ_instab * Linstab
    return L, (; Lce, KL, recL, Lsq, Lneg, Lself, Ljnd, Linstab)
end

function lossfn(m::SemioticModel, sequence::AbstractVector{<:Integer}; kwargs...)
    context, targets = next_token_pairs(sequence)
    return lossfn(m, context, targets; kwargs...)
end

function lossfn(m::SemioticModel, sequences::AbstractMatrix{<:Integer}; kwargs...)
    context, targets = next_token_pairs(sequences)
    return lossfn(m, context, targets; kwargs...)
end

# -----------------------------
# Toy usage
# -----------------------------
function toy_train(; seed=7)
    Random.seed!(seed)
    vocab = 8; d = 64
    sq = SemioticSquare(1, 2, 3, 4)
    m = SemioticModel(vocab, d; layers=2, H=3, k=8, z=32, square=sq)

    tokens  = [1, 2, 5, 6, 7, 3, 4, 2]
    targets = [2, 1, 5, 6, 7, 4, 3, 1]
    opt = Flux.setup(Flux.Optimisers.Adam(1e-3), m)

    for step in 1:80
        gs = Flux.gradient(Flux.params(m)) do
            L, _ = lossfn(m, tokens, targets; λ_square=0.05f0, λ_neg=0.01f0)
            L
        end
        Flux.update!(opt, Flux.params(m), gs)
        _ = forward(m, tokens; update_field=true, will=true)
        if step % 10 == 0
            L, parts = lossfn(m, tokens, targets; λ_square=0.05f0, λ_neg=0.01f0)
            @info "step=$step" loss=L parts
        end
    end
    return m
end

# =============================
# Archetypal subcategories (V4)
# =============================
module Archetypal

using Flux, LinearAlgebra, Random, NNlib, Functors
import ..SemioticTransformer
using ..SemioticTransformer: T, Negation, negation_penalty, DifferenceField, difference_matrix,
    MeaningField, potential, potential_grad, update!, MeaningChainLayer, SelfField, coniunctio,
    _apply_layernorm, _apply_dense, next_token_pairs, _ce_loss

struct Obj
    d::Int
end

struct Morphism
    dom::Obj
    cod::Obj
    A::Matrix{T}
end
@functor Morphism

id(o::Obj) = Morphism(o, o, Matrix{T}(I, o.d, o.d))

compose(g::Morphism, f::Morphism) =
    (f.cod.d == g.dom.d) ? Morphism(f.dom, g.cod, g.A * f.A) :
    throw(DimensionMismatch("compose: cod(f)!=dom(g)"))

function category_penalty(f::Morphism, g::Morphism, h::Morphism; λ_assoc=T(1e-3), λ_id=T(1e-3))
    lhs = compose(compose(h, g), f).A
    rhs = compose(h, compose(g, f)).A
    L_assoc = norm(lhs .- rhs)^2
    I_dom = id(f.dom).A
    I_cod = id(f.cod).A
    L_id = norm(f.A - f.A * I_dom)^2 + norm(f.A - I_cod * f.A)^2
    λ_assoc * L_assoc + λ_id * L_id
end

struct Functor
    U::Matrix{T}
    V::Matrix{T}
end
@functor Functor

functor_map(F::Functor, f::Morphism) =
    Morphism(Obj(size(F.U, 1)), Obj(size(F.U, 1)), F.U * f.A * F.V)

function functor_penalty(F::Functor, f::Morphism, g::Morphism; λ=T(1e-3))
    lhs = functor_map(F, compose(g, f)).A
    rhs = functor_map(F, g).A * functor_map(F, f).A
    λ * norm(lhs .- rhs)^2
end

@enum HeadRole::UInt8 begin
    Contradiction = 1
    Contrary = 2
    Implication = 3
end

struct SemioticHead
    WQ::Matrix{T}
    WK::Matrix{T}
    WV::Matrix{T}
    df::DifferenceField
    β::T
    γ::T
    η::T
    role::HeadRole
end
@functor SemioticHead

function SemioticHead(d::Int, dk::Int, dv::Int, role::HeadRole; β=T(0.2), γ=T(0.2), η=T(0.2))
    SemioticHead(
        Flux.glorot_uniform(dk, d),
        Flux.glorot_uniform(dk, d),
        Flux.glorot_uniform(dv, d),
        DifferenceField(dk),
        β,
        γ,
        η,
        role,
    )
end

struct SemioticMHA
    heads::Vector{SemioticHead}
    WO::Matrix{T}
    neg::Negation
end
@functor SemioticMHA

function SemioticMHA(d::Int; H::Int=3, dk::Int=div(d, H), dv::Int=div(d, H))
    roles = [Contradiction, Contrary, Implication][1:H]
    hs = [
        SemioticHead(d, dk, dv, r;
            β = r == Contradiction ? T(0.8) : r == Contrary ? T(0.4) : T(0.2),
            γ = r == Implication ? T(0.6) : T(0.2),
            η = T(0.3),
        ) for r in roles
    ]
    SemioticMHA(hs, Flux.glorot_uniform(d, dv * H), Negation(d))
end

neg_penalty(mha::SemioticMHA; kw...) = negation_penalty(mha.neg; kw...)

function role_bias(h::SemioticHead, Q::AbstractMatrix{T}, K::AbstractMatrix{T}, Φ::AbstractVector{T}, neg::Negation)
    n = size(Q, 2)
    Φrow = reshape(Φ, 1, :)
    Φpair = (Φrow .+ Φrow') .* 0.5f0
    S = (Q' * K) ./ sqrt(T(size(Q, 1)))
    D = difference_matrix(h.df, K)
    R = zeros(T, n, n)
    if h.role == Implication
        NK = negate(neg, K)
        S_imp = (Q' * NK) ./ sqrt(T(size(Q, 1)))
        R .= h.η .* S_imp
    elseif h.role == Contradiction
        R .= -h.η .* D
    elseif h.role == Contrary
        R .= -0.5f0 .* h.η .* D
    end
    S .- h.β .* D .+ h.γ .* Φpair .+ R
end

function (mha::SemioticMHA)(X::AbstractMatrix{T}, Φ::AbstractVector{T})
    parts = Matrix{T}[]
    for h in mha.heads
        Q, K, V = h.WQ * X, h.WK * X, h.WV * X
        A = role_bias(h, Q, K, Φ, mha.neg)
        P = NNlib.softmax(A; dims=2)
        push!(parts, V * P')
    end
    mha.WO * reduce(vcat, parts)
end

struct ArchetypeUnit
    tag::UInt8
    ds::Int
    neg::Negation
    df::DifferenceField
    mf::MeaningField
    mha::SemioticMHA
    norm1::LayerNorm
    chain::MeaningChainLayer
    norm2::LayerNorm
    sf::SelfField
    F::Functor
    morphs::Vector{Morphism}
end
@functor ArchetypeUnit

function ArchetypeUnit(tag::UInt8, d::Int, ds::Int; k::Int=6, H::Int=3)
    obj = Obj(ds)
    morphs = [Morphism(obj, obj, Flux.glorot_uniform(ds, ds)) for _ in 1:3]
    ArchetypeUnit(
        tag,
        ds,
        Negation(ds),
        DifferenceField(ds),
        MeaningField(ds, k),
        SemioticMHA(ds; H=H),
        Flux.LayerNorm(ds),
        MeaningChainLayer(ds),
        Flux.LayerNorm(ds),
        SelfField(ds),
        Functor(Flux.glorot_uniform(d, ds), Flux.glorot_uniform(ds, d)),
        morphs,
    )
end

"""Return local codes and global projection for a single sequence."""
function _unit_forward(u::ArchetypeUnit, Xg::AbstractMatrix{T}; will::Bool=true, update_field::Bool=false)
    X = u.F.V * Xg
    Φ = potential(u.mf, X)
    Y = X .+ u.mha(u.norm1(X), Φ)
    Z = Y .+ u.chain(u.norm2(Y), Φ)
    if will
        Gφ = similar(Z)
        for j in 1:size(Z, 2)
            Gφ[:, j] = potential_grad(u.mf, view(Z, :, j))
        end
        GD = zeros(T, size(Z, 1), size(Z, 2))
        μ = mean(Z; dims=2)
        @inbounds for j in 1:size(Z, 2)
            GD[:, j] .= 2f0 .* (u.df.M * (size(Z, 2) .* view(Z, :, j) .- vec(μ)))
        end
        Z .+= 1e-2f0 .* Gφ .- 1e-3f0 .* GD
    end
    if update_field
        update!(u.mf, Z)
    end
    return Z, u.F.U * Z
end

"""Return local codes and global projection for batched sequences (d×n×b)."""
function _unit_forward(u::ArchetypeUnit, Xg::AbstractArray{T,3}; will::Bool=true, update_field::Bool=false)
    batches = size(Xg, 3)
    locals = Array{T}(undef, u.ds, size(Xg, 2), batches)
    globals = Array{T}(undef, size(u.F.U, 1), size(Xg, 2), batches)
    @inbounds @views for b in 1:batches
        loc, glob = _unit_forward(u, view(Xg, :, :, b); will=will, update_field=update_field)
        locals[:, :, b] .= loc
        globals[:, :, b] .= glob
    end
    return locals, globals
end

function (u::ArchetypeUnit)(Xg::AbstractMatrix{T}; will::Bool=true, update_field::Bool=false)
    _, Y = _unit_forward(u, Xg; will=will, update_field=update_field)
    return Y
end

function (u::ArchetypeUnit)(Xg::AbstractArray{T,3}; will::Bool=true, update_field::Bool=false)
    _, Y = _unit_forward(u, Xg; will=will, update_field=update_field)
    return Y
end

center_global(u::ArchetypeUnit) = u.F.U * u.sf.s

struct ArchetypeRouter
    gate::Dense
    τ::T
end
@functor ArchetypeRouter

ArchetypeRouter(d::Int, K::Int; τ=T(1.0)) = ArchetypeRouter(Dense(d, K), τ)

route(ar::ArchetypeRouter, X::AbstractMatrix{T}) = NNlib.softmax(ar.gate(X) ./ ar.τ; dims=1)
function route(ar::ArchetypeRouter, X::AbstractArray{T,3})
    K = size(ar.gate.weight, 1)
    n = size(X, 2)
    batches = size(X, 3)
    W = Array{T}(undef, K, n, batches)
    @inbounds @views for b in 1:batches
        W[:, :, b] = route(ar, view(X, :, :, b))
    end
    return W
end

"""Scene monoid for pairwise archetype composition (Hadamard)."""
struct SceneMonoid
    U::Matrix{T}            # d×r
    P::Vector{Matrix{T}}    # K×(r×ds) projections
    e::Vector{T}            # unit element (≈1)
    λ_pair::T
    pairs::Vector{Tuple{Int, Int}}
end
@functor SceneMonoid

function SceneMonoid(d::Int, ds::Int, K::Int, r::Int; λ_pair=T(0.5), allowed_pairs::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing)
    raw_pairs = isnothing(allowed_pairs) ? [(a, b) for a in 1:K-1 for b in (a + 1):K] :
        [(min(a, b), max(a, b)) for (a, b) in allowed_pairs if 1 <= a <= K && 1 <= b <= K && a != b]
    uniq = sort!(collect(Set(raw_pairs)); by=x -> (x[1], x[2]))
    isempty(uniq) && (uniq = [(a, b) for a in 1:K-1 for b in (a + 1):K])
    SceneMonoid(Flux.glorot_uniform(d, r), [Flux.glorot_uniform(r, ds) for _ in 1:K], ones(T, r), λ_pair, uniq)
end

"""Compose pairwise scene interactions for a single sequence."""
function scene_compose(sm::SceneMonoid, locals::Vector{<:AbstractMatrix{T}}, W::AbstractMatrix{T})
    K = length(locals)
    rdim = size(sm.U, 2)
    n = size(W, 2)
    R = [sm.P[k] * locals[k] for k in 1:K]  # r×n projections
    rsum = zeros(T, rdim, n)
    @inbounds for i in 1:n
        wi = view(W, :, i)
        for (a, b) in sm.pairs
            wpair = wi[a] * wi[b]
            if wpair > 0
                rsum[:, i] .+= wpair .* (view(R[a], :, i) .* view(R[b], :, i))
            end
        end
    end
    sm.λ_pair .* (sm.U * rsum)
end

"""Compose pairwise scene interactions for batched sequences."""
function scene_compose(sm::SceneMonoid, locals::Vector{<:AbstractArray{T,3}}, W::AbstractArray{T,3})
    K = length(locals)
    rdim = size(sm.U, 2)
    n = size(W, 2)
    batches = size(W, 3)
    Y = zeros(T, size(sm.U, 1), n, batches)
    @inbounds @views for b in 1:batches
        R = [sm.P[k] * view(locals[k], :, :, b) for k in 1:K]
        rsum = zeros(T, rdim, n)
        for i in 1:n
            wi = view(W, :, i, b)
            for (a, c) in sm.pairs
                wpair = wi[a] * wi[c]
                if wpair > 0
                    rsum[:, i] .+= wpair .* (view(R[a], :, i) .* view(R[c], :, i))
                end
            end
        end
        Y[:, :, b] .= sm.λ_pair .* (sm.U * rsum)
    end
    Y
end

function monoid_penalty(sm::SceneMonoid, locals::Vector{<:AbstractMatrix{T}}; λ_unit=T(1e-3), λ_assoc=T(1e-3), samples::Int=4)
    K = length(locals)
    n = size(locals[1], 2)
    rdim = size(sm.U, 2)
    sample_ids = 1:min(samples, n)
    L_unit = zero(T)
    for t in sample_ids, u in 1:K
        r = sm.P[u] * view(locals[u], :, t:t)
        L_unit += mean((r .* sm.e .- r).^2)
    end
    L_assoc = zero(T)
    if K >= 3
        for t in sample_ids
            ra = sm.P[1] * view(locals[1], :, t:t)
            rb = sm.P[2] * view(locals[2], :, t:t)
            rc = sm.P[3] * view(locals[3], :, t:t)
            L_assoc += mean(((ra .* rb) .* rc .- (ra .* (rb .* rc))).^2)
        end
    end
    λ_unit * L_unit + λ_assoc * L_assoc
end

function monoid_penalty(sm::SceneMonoid, locals::Vector{<:AbstractArray{T,3}}; λ_unit=T(1e-3), λ_assoc=T(1e-3), samples::Int=4)
    K = length(locals)
    n = size(locals[1], 2)
    batches = size(locals[1], 3)
    sample_ids = 1:min(samples, n)
    L_unit = zero(T)
    for b in 1:batches
        for t in sample_ids, u in 1:K
            r = sm.P[u] * view(locals[u], :, t:t, b)
            L_unit += mean((r .* sm.e .- r).^2)
        end
    end
    L_assoc = zero(T)
    if K >= 3
        for b in 1:batches, t in sample_ids
            ra = sm.P[1] * view(locals[1], :, t:t, b)
            rb = sm.P[2] * view(locals[2], :, t:t, b)
            rc = sm.P[3] * view(locals[3], :, t:t, b)
            L_assoc += mean(((ra .* rb) .* rc .- (ra .* (rb .* rc))).^2)
        end
    end
    λ_unit * L_unit + λ_assoc * L_assoc
end

struct ArchetypalBlock
    units::Vector{ArchetypeUnit}
    router::ArchetypeRouter
    monoid::SceneMonoid
    df_global::DifferenceField
    neg_global::Negation
end
@functor ArchetypalBlock

function ArchetypalBlock(d::Int; K::Int=6, ds::Int=div(d, 2), r::Int=32, λ_pair::T=0.5f0,
        allowed_pairs::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing)
    units = [ArchetypeUnit(UInt8(i), d, ds) for i in 1:K]
    ArchetypalBlock(
        units,
        ArchetypeRouter(d, K),
        SceneMonoid(d, ds, K, r; λ_pair=λ_pair, allowed_pairs=allowed_pairs),
        DifferenceField(d),
        Negation(d),
    )
end

function (ab::ArchetypalBlock)(X::AbstractMatrix{T}; will::Bool=true, update_fields::Bool=false)
    W = route(ab.router, X)
    Y = zeros(T, size(X, 1), size(X, 2))
    locals = Vector{Matrix{T}}(undef, length(ab.units))
    for (k, u) in enumerate(ab.units)
        locals[k], Yk = _unit_forward(u, X; will=will, update_field=update_fields)
        Y .+= Yk .* reshape(W[k, :], 1, :)
    end
    scene = scene_compose(ab.monoid, locals, W)
    return Y .+ scene, W, (; locals)
end

function (ab::ArchetypalBlock)(X::AbstractArray{T,3}; will::Bool=true, update_fields::Bool=false)
    _, n, batches = size(X)
    K = length(ab.units)
    W = Array{T}(undef, K, n, batches)
    Y = zeros(T, size(X)...)
    locals = Vector{Array{T,3}}(undef, K)
    @inbounds @views for b in 1:batches
        W[:, :, b] = route(ab.router, view(X, :, :, b))
    end
    for (k, u) in enumerate(ab.units)
        locals[k], Yk = _unit_forward(u, X; will=will, update_field=update_fields)
        @inbounds @views for b in 1:batches
            Y[:, :, b] .+= view(Yk, :, :, b) .* reshape(W[k, :, b], 1, :)
        end
    end
    scene = scene_compose(ab.monoid, locals, W)
    return Y .+ scene, W, (; locals)
end

function archetype_rules_loss(ab::ArchetypalBlock; λ_self=T(1e-2), λ_pair=T(1e-2), margin=T(1.4))
    K = length(ab.units)
    c = [center_global(u) for u in ab.units]
    if K < 3
        return zero(T)
    end
    Lself = norm(c[1] .- coniunctio(ab.neg_global, c[2]; α=T(0.5)))^2
    Lps = norm(c[3] .- (ab.neg_global.N * c[2]))^2
    Lan = (K >= 5) ? norm(c[4] .- (ab.neg_global.N * c[5]))^2 : zero(T)
    Lht = (K >= 6) ? NNlib.relu(margin .- norm(c[6] .- c[min(2, K)]))^2 : zero(T)
    λ_self * Lself + λ_pair * (Lps + Lan + Lht)
end

function jnd_loss(df::DifferenceField, X::AbstractMatrix{T}; k=T(0.08), θ=T(0.05))
    d, n = size(X)
    L = zero(T)
    for j in 1:n
        x = view(X, :, j)
        δ = k * max(norm(x), T(1.0)) .* randn(T, d) ./ sqrt(T(d))
        xp = x .+ δ
        Dloc = dot((x - xp), df.M * (x - xp)) |> df.φ
        L += (Dloc - θ)^2
    end
    L / n
end

function jnd_loss(df::DifferenceField, X::AbstractArray{T,3}; k=T(0.08), θ=T(0.05))
    batches = size(X, 3)
    acc = zero(T)
    @inbounds @views for b in 1:batches
        acc += jnd_loss(df, view(X, :, :, b); k=k, θ=θ)
    end
    return acc / batches
end

struct ArchetypalModel
    emb::Embedding
    block::ArchetypalBlock
    ln::LayerNorm
    proj::Dense
end
@functor ArchetypalModel

function ArchetypalModel(vocab::Int, d::Int; K::Int=6, ds::Int=div(d, 2), r::Int=32, λ_pair::T=0.5f0, classes::Int=vocab,
        allowed_pairs::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing)
    ArchetypalModel(
        Flux.Embedding(vocab, d),
        ArchetypalBlock(d; K=K, ds=ds, r=r, λ_pair=λ_pair, allowed_pairs=allowed_pairs),
        Flux.LayerNorm(d),
        Dense(d, classes),
    )
end

function forward(m::ArchetypalModel, tokens::AbstractVector{<:Integer}; update_fields::Bool=false, will::Bool=true)
    X = m.emb(tokens)
    Y, W, cache = m.block(X; will=will, update_fields=update_fields)
    Y = _apply_layernorm(m.ln, Y)
    logits = m.proj(Y)
    return logits, Y, W, cache
end

function forward(m::ArchetypalModel, tokens::AbstractMatrix{<:Integer}; update_fields::Bool=false, will::Bool=true)
    X = m.emb(tokens)
    Y, W, cache = m.block(X; will=will, update_fields=update_fields)
    Y = _apply_layernorm(m.ln, Y)
    logits = _apply_dense(m.proj, Y)
    return logits, Y, W, cache
end

function structure_penalty(ab::ArchetypalBlock)
    L = zero(T)
    for u in ab.units
        if length(u.morphs) >= 2
            f, g = u.morphs[1:2]
            L += category_penalty(f, g, compose(g, f); λ_assoc=T(1e-3), λ_id=T(1e-3))
            L += functor_penalty(u.F, f, g; λ=T(1e-3))
        end
        L += neg_penalty(u.mha; λ_inv=T(1e-3), λ_iso=T(1e-3), λ_sym=T(1e-4))
    end
    L += negation_penalty(ab.neg_global; λ_inv=T(5e-4), λ_iso=T(5e-4), λ_sym=T(1e-4))
    L
end

function lossfn(m::ArchetypalModel, tokens::AbstractVector{<:Integer}; pad_token::Union{Nothing,Int}=nothing,
        λ_struct=T(1e-3), λ_rules=T(1e-2), λ_jnd=T(1e-3), λ_mono=T(1e-3), update_fields::Bool=false, will::Bool=true)
    context, targets = next_token_pairs(tokens)
    return lossfn(m, context, targets; pad_token=pad_token, λ_struct=λ_struct, λ_rules=λ_rules, λ_jnd=λ_jnd,
        λ_mono=λ_mono, update_fields=update_fields, will=will)
end

function lossfn(m::ArchetypalModel, tokens::AbstractMatrix{<:Integer}; pad_token::Union{Nothing,Int}=nothing,
        λ_struct=T(1e-3), λ_rules=T(1e-2), λ_jnd=T(1e-3), λ_mono=T(1e-3), update_fields::Bool=false, will::Bool=true)
    context, targets = next_token_pairs(tokens)
    return lossfn(m, context, targets; pad_token=pad_token, λ_struct=λ_struct, λ_rules=λ_rules, λ_jnd=λ_jnd,
        λ_mono=λ_mono, update_fields=update_fields, will=will)
end

function lossfn(m::ArchetypalModel, tokens::AbstractVector{<:Integer}, targets::AbstractVector{<:Integer};
        pad_token::Union{Nothing,Int}=nothing, λ_struct=T(1e-3), λ_rules=T(1e-2), λ_jnd=T(1e-3), λ_mono=T(1e-3),
        update_fields::Bool=false, will::Bool=true)
    logits, Y, _, cache = forward(m, tokens; update_fields=update_fields, will=will)
    Lce = _ce_loss(logits, targets; pad_token=pad_token)
    Lstruct = structure_penalty(m.block)
    Lrules = archetype_rules_loss(m.block)
    Ljnd = jnd_loss(m.block.df_global, Y)
    Lmono = monoid_penalty(m.block.monoid, cache.locals)
    L = Lce + λ_struct * Lstruct + λ_rules * Lrules + λ_jnd * Ljnd + λ_mono * Lmono
    return L, (; Lce, Lstruct, Lrules, Ljnd, Lmono)
end

function lossfn(m::ArchetypalModel, tokens::AbstractMatrix{<:Integer}, targets::AbstractMatrix{<:Integer};
        pad_token::Union{Nothing,Int}=nothing, λ_struct=T(1e-3), λ_rules=T(1e-2), λ_jnd=T(1e-3), λ_mono=T(1e-3),
        update_fields::Bool=false, will::Bool=true)
    logits, Y, _, cache = forward(m, tokens; update_fields=update_fields, will=will)
    Lce = _ce_loss(logits, targets; pad_token=pad_token)
    Lstruct = structure_penalty(m.block)
    Lrules = archetype_rules_loss(m.block)
    Ljnd = jnd_loss(m.block.df_global, Y)
    Lmono = monoid_penalty(m.block.monoid, cache.locals)
    L = Lce + λ_struct * Lstruct + λ_rules * Lrules + λ_jnd * Ljnd + λ_mono * Lmono
    return L, (; Lce, Lstruct, Lrules, Ljnd, Lmono)
end

function toy_train(; seed=2025)
    Random.seed!(seed)
    vocab = 12
    d = 64
    K = 6
    ds = 32
    r = 32
    m = ArchetypalModel(vocab, d; K=K, ds=ds, r=r)

    tokens = [1, 2, 5, 6, 7, 3, 4, 2, 9, 10]
    targets = [2, 1, 5, 6, 7, 4, 3, 1, 10, 9]

    opt = Flux.setup(Flux.Optimisers.Adam(1e-3), m)

    for step in 1:80
        gs = Flux.gradient(Flux.params(m)) do
            L, _ = lossfn(m, tokens, targets)
            L
        end
        Flux.update!(opt, Flux.params(m), gs)
        forward(m, tokens; update_fields=true)

        if step % 10 == 0
            L, parts = lossfn(m, tokens, targets)
            @info "step=$step" loss=L parts
        end
    end
    m
end

end # module Archetypal

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running toy training loop from SemioticTransformer.jl"
    SemioticTransformer.toy_train()
end
