vocab_size = 50_000
n_positions = 256
n_embd = 32
n_layer = 4

total_params = (
    vocab_size * n_embd
    + n_positions * n_embd
    + n_layer * (12 * n_embd * n_embd + 13 * n_embd)
    + 2 * n_embd
)
print(f"Total parameters: {total_params:,}")
