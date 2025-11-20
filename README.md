# Semiotic Transformer

A compact Julia implementation of a "semiotic" Transformer that injects ideas from
structural linguistics and semiotics into an otherwise minimal Transformer block.
The module is kept in a single file (`SemioticTransformer.jl`) so that it can be read
like a Karpathy-style mini model while still exposing new concepts such as
Difference/Meaning fields, Greimas semiotic squares, and a multi-path meaning chain.

## Highlights

* **DifferenceField** – computes a smooth quadratic notion of "difference" between
  token representations and injects it into the attention score.
* **MeaningField** – acts like an attractor potential that biases both the attention
  mechanism and the meaning-chain recomposition.
* **SemioticSquare constraints** – optional metric regularisers over vocabulary
  slots that keep A/¬A/B/¬B relationships coherent. You can attach multiple squares
  at once (pass `squares=[...]` when building the model). When enabled you can also
  learn a linear **Negation** operator that behaves like an involution and maps
  A → ¬A.
* **MeaningChainLayer** – separates denotation, connotation, and mythic pathways and
  recombines them with a learned softmax gate.

## Quick start

The module does not ship with a package `Project.toml`; simply include the file and
use the module directly:

```julia
julia> include("SemioticTransformer.jl"); using .SemioticTransformer
```

You can then run the toy optimiser step:

```julia
julia> SemioticTransformer.toy_train()
```

By default the toy example wires a SemioticSquare over the first four vocabulary
indices and enables the Negation operator. The helper `lossfn(model, sequence)` will
shift the sequence internally (tokens `1:n-1` predict `2:n`), add the summed square
loss across all registered squares, and apply the negation regulariser.

If your sequences contain padding, pass the pad token through to ignore those
positions during the cross-entropy term:

```julia
loss = SemioticTransformer.lossfn(model, sequence; pad_token=0)
```

If you want manual control over the training pairs you can call the more explicit
method:

```julia
context, targets = SemioticTransformer.next_token_pairs(sequence)
loss = SemioticTransformer.lossfn(model, context, targets; λ_square=0.05f0, λ_neg=0.01f0)
```

### Mini-batching

Every forward/loss helper also accepts a matrix of tokens whose columns represent
different sequences (shape `sequence_length × batch`). The `next_token_pairs`
utility and the loss function will shift each column independently and the model
returns logits shaped `classes × sequence_length × batch`.

```julia
sequences = hcat([1, 2, 5, 6], [2, 3, 4, 1])
loss = SemioticTransformer.lossfn(model, sequences; λ_square=0.05f0)
```

When columns have padding (e.g. ragged sentences packed together), supply
`pad_token` so the masked positions are dropped from the cross-entropy term while
still flowing through the semiotic penalties:

```julia
sequences = hcat([1, 2, 5, 6], [2, 3, 0, 0])
loss = SemioticTransformer.lossfn(model, sequences; pad_token=0)
```

To supervise more than one semiotic relation simultaneously, pass a collection of
`SemioticSquare`s via the `squares` keyword (or the legacy `square` for a single
constraint):

```julia
squares = [SemioticSquare(1, 2, 3, 4), SemioticSquare(5, 6, 7, 8)]
model = SemioticModel(vocab, d; squares=squares, use_negation=true)
```

## Notes

* The implementation uses `@views` and exploits the symmetry of the difference
  matrix so the cost of the DifferenceField stays manageable even for modest
  sequence lengths.
* The Negation operator is part of the model parameters when `use_negation=true` and
  is jointly optimised with the rest of the network.
* Everything is written in Flux-friendly style, so gradients flow through the
  Difference/Meaning fields, attention, and the semiotic constraints automatically.
