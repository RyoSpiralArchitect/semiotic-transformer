module SemioticTransformer

using Flux
using LinearAlgebra
using Statistics
using Random
using NNlib # for softplus, softmax
using Functors
const T = Float32

# -----------------------------
# Core objects
# -----------------------------
abstract type SemioticObject end

"概念ベクトル（d×n表現で列がトークン）"
struct Concept <: SemioticObject
    # nothing; Conceptは単なるベクトル列Xで運ぶ
end

"否定（involution）: x ↦ N x"
struct Negation
    N::Matrix{T} # d×d
end
@functor Negation
Negation(d::Int; init=Flux.glorot_uniform) = Negation(init(d, d))
negate(neg::Negation, x::AbstractVector{T}) = neg.N * x

# -----------------------------
# Difference field: pairwise差異場 D(i,j)
# -----------------------------
struct DifferenceField
    M::Matrix{T}   # d×d 正定値を想定（学習可）
    φ::Function    # 非線形（softplusなど）
end
@functor DifferenceField
DifferenceField(d::Int; φ = NNlib.softplus) = DifferenceField(Matrix{T}(I, d, d), φ)

"X: d×n（列がトークン）。D: n×n の差異行列。"
function difference_matrix(df::DifferenceField, X::AbstractMatrix{T})
    d, n = size(X)
    M = df.M
    D = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n
        xi = view(X, :, i)
        for j in 1:n
            Δ = xi .- view(X, :, j)
            D[i, j] = dot(Δ, M * Δ)
        end
    end
    return df.φ.(D)
end

# -----------------------------
# Meaning field: ポテンシャルΦとその勾配
# -----------------------------
"意味プロトタイプ p_k を置き、Φ(x) = scale * Σ_k -w_k ||x - p_k||^2"
struct MeaningField
    P::Matrix{T}   # d×k
    w::Vector{T}   # k
    scale::T
end
@functor MeaningField
MeaningField(d::Int, k::Int; scale=T(1.0)) =
    MeaningField(Flux.glorot_uniform(d, k), ones(T, k), scale)

"X: d×n → Φ: n（各列のポテンシャル）"
function potential(mf::MeaningField, X::AbstractMatrix{T})
    d, n = size(X); k = size(mf.P, 2)
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

# （必要なら）Φの「勾配」をAttentionに反映したければ、Zygoteが自動で∂L/∂P, ∂L/∂wまで流す。
# さらに明示的な場の勾配 ∇Φ(x) が欲しければ以下を使う：
function potential_grad(mf::MeaningField, x::AbstractVector{T})
    # ∇_x Φ = scale * Σ_k -w_k * 2 (x - p_k)
    g = zeros(T, length(x))
    @inbounds for t in 1:size(mf.P, 2)
        g .+= -2f0 * mf.w[t] .* (x .- view(mf.P, :, t))
    end
    return mf.scale .* g
end

# -----------------------------
# Semiotic square（Greimas）
# -----------------------------
"語彙インデックスで四角形を持つ（列位置）"
struct SemioticSquare
    s1::Int   # A
    s2::Int   # B
    ns1::Int  # ¬A
    ns2::Int  # ¬B
end

"四角形の関係制約を満たすよう距離に基づく損失を構成（hinge）"
function square_loss(X::AbstractMatrix{T}, sq::SemioticSquare; margin_contra=T(2.0), margin_imp=T(0.5))
    # X: d×n から関係点を抜く
    A  = view(X, :, sq.s1); nA = view(X, :, sq.ns1)
    B  = view(X, :, sq.s2); nB = view(X, :, sq.ns2)

    # 距離
    d(x,y) = norm(x .- y)

    # 矛盾: A vs ¬A, B vs ¬B は遠く（> margin_contra）
    L_contra = NNlib.relu(margin_contra .- d(A, nA))^2 + NNlib.relu(margin_contra .- d(B, nB))^2

    # 反対（contrary）: A vs B, ¬A vs ¬B もそこそこ遠く
    L_contrary = 0.5f0 * (NNlib.relu(margin_contra .- d(A, B))^2 + NNlib.relu(margin_contra .- d(nA, nB))^2)

    # 含意（implication）: A→¬B, B→¬A は近め（< margin_imp）
    L_impl = (NNlib.relu(d(A, nB) .- margin_imp)^2 + NNlib.relu(d(B, nA) .- margin_imp)^2)

    return L_contra + L_contrary + L_impl
end

# -----------------------------
# Meaning chain layer（Barthes-ish）
# -----------------------------
"denotation/connotation/myth の3経路 + Φで重み付け"
struct MeaningChainLayer
    den::Dense
    con::Dense
    myth::Dense
    α::Vector{T}  # 3要素のゲート係数
    act::Function
end
@functor MeaningChainLayer
function MeaningChainLayer(d::Int; h::Int=d, act = gelu)
    MeaningChainLayer(Dense(d, d), Dense(d, h, act), Dense(h, d), param([T(0.6), T(0.3), T(0.1)]), act)
end

"X: d×n, Φ: n → Y: d×n"
function (m::MeaningChainLayer)(X::AbstractMatrix{T}, Φ::AbstractVector{T})
    d, n = size(X)
    Φrow = reshape(Φ, 1, :)
    # 基本の3経路
    d0 = m.den(X)                           # 直写（denotation）
    c0 = m.con(X) .* (1 .+ Φrow)            # 文脈で増幅（connotation）
    y0 = m.myth(c0)                          # 物語的再符号化（myth）
    # ゲート合成
    α = NNlib.softmax(m.α)
    return α[1] .* d0 .+ α[2] .* c0 .+ α[3] .* y0
end

# -----------------------------
# Semiotic attention
# -----------------------------
"相関 − β·差異 + γ·(Φ_i + Φ_j)/2 でスコア形成"
struct SemioticAttention
    WQ::Matrix{T}  # dk×d
    WK::Matrix{T}  # dk×d
    WV::Matrix{T}  # dv×d
    WO::Matrix{T}  # d×dv
    df::DifferenceField
    β::T
    γ::T
end
@functor SemioticAttention
function SemioticAttention(d::Int, dk::Int; dv::Int=d, β::T=T(0.2), γ::T=T(0.2))
    SemioticAttention(Flux.glorot_uniform(dk, d),
                      Flux.glorot_uniform(dk, d),
                      Flux.glorot_uniform(dv, d),
                      Flux.glorot_uniform(d, dv),
                      DifferenceField(dk),
                      β, γ)
end

"X: d×n, Φ: n → Y: d×n"
function (sa::SemioticAttention)(X::AbstractMatrix{T}, Φ::AbstractVector{T})
    # 線形射
    Q = sa.WQ * X   # dk×n
    K = sa.WK * X   # dk×n
    V = sa.WV * X   # dv×n
    dk = size(Q, 1)

    # 通常相関
    S = (Q' * K) ./ sqrt(T(dk))  # n×n（query×key）

    # 差異場（keys空間で）
    D = difference_matrix(sa.df, K)  # n×n

    # 意味ポテンシャル（i,j）→ (Φ_i + Φ_j)/2
    Φrow = reshape(Φ, 1, :)
    Φpair = (Φrow .+ Φrow') .* 0.5f0  # n×n

    # スコア合成
    A = S .- sa.β .* D .+ sa.γ .* Φpair

    # 正規化＆適用
    P = NNlib.softmax(A; dims=2)       # 各query行でsoftmax
    Yv = V * P'                        # dv×n
    return sa.WO * Yv                  # d×n
end

# -----------------------------
# Transformer block（semiotic）
# -----------------------------
struct SemioticBlock
    attn::SemioticAttention
    norm1::LayerNorm
    chain::MeaningChainLayer
    norm2::LayerNorm
    mf::MeaningField
end
@functor SemioticBlock
function SemioticBlock(d::Int; dk::Int=div(d,2), h::Int=d, k::Int=8)
    SemioticBlock(
        SemioticAttention(d, dk),
        Flux.LayerNorm(d),
        MeaningChainLayer(d; h=h),
        Flux.LayerNorm(d),
        MeaningField(d, k)
    )
end

"X: d×n → d×n"
function (b::SemioticBlock)(X::AbstractMatrix{T})
    Φ = potential(b.mf, X)
    Y = X .+ b.attn(b.norm1(X), Φ)
    Z = Y .+ b.chain(b.norm2(Y), Φ)
    return Z
end

# -----------------------------
# Tiny model
# -----------------------------
"エンベッディング + SemioticBlock × L"
struct SemioticModel
    emb::Embedding
    blocks::Vector{SemioticBlock}
    ln::LayerNorm
    proj::Dense
    square::Union{Nothing, SemioticSquare}
end
@functor SemioticModel

function SemioticModel(vocab::Int, d::Int; layers::Int=2, dk::Int=div(d,2), h::Int=d, k::Int=8, classes::Int=vocab, square::Union{Nothing,SemioticSquare}=nothing)
    blocks = [SemioticBlock(d; dk=dk, h=h, k=k) for _ in 1:layers]
    SemioticModel(Flux.Embedding(vocab, d), blocks, LayerNorm(d), Dense(d, classes), square)
end

"tokens: n（Int）→ logits: classes×n"
function (m::SemioticModel)(tokens::AbstractVector{Int})
    X = m.emb(tokens)             # d×n
    for b in m.blocks
        X = b(X)                  # d×n
    end
    X = m.ln(X)                   # d×n
    return m.proj(X)              # classes×n
end

# -----------------------------
# Loss (CE + square制約)
# -----------------------------
function lossfn(m::SemioticModel, tokens::Vector{Int}, targets::Vector{Int}; λ_square = T(0.1))
    logits = m(tokens)                      # C×n
    Lce = Flux.logitcrossentropy(logits, onehotbatch(targets, 1:size(logits,1)))
    Lsq = (m.square === nothing) ? 0f0 : square_loss(m.emb.weight, m.square) # ここでは語彙エンベ上に制約を入れる例
    return Lce + λ_square * Lsq
end

# -----------------------------
# Toy usage (run manually)
# -----------------------------
"""
toy_train():
  - 語彙 8、隠れ次元 d=64 のモデルを作成
  - 適当データで1ステップ学習（使い方の見本）
"""
function toy_train(; seed=42)
    Random.seed!(seed)
    vocab = 8; d=64
    # 例: 1:hot, 2:cold, 3:not_hot, 4:not_cold を四角形に
    sq = SemioticSquare(1, 2, 3, 4)
    m = SemioticModel(vocab, d; layers=2, square=sq)

    tokens  = [1,2,5,6,7,3,4,2]               # n=8
    targets = [2,1,5,6,7,4,3,1]               # 適当な次トークン予測風
    opt = Flux.setup(Flux.Optimisers.Adam(1e-3), m)

    gs = Flux.gradient(Flux.params(m)) do
        lossfn(m, tokens, targets; λ_square=0.05f0)
    end
    Flux.update!(opt, Flux.params(m), gs)

    @info "Step OK" loss = lossfn(m, tokens, targets; λ_square=0.05f0)
    return m
end

end # module
