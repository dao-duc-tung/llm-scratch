vocab_size = 50_000
n_positions = 1024
n_embd = 8
n_layer = 2

total_params = (
    vocab_size * n_embd
    + n_positions * n_embd
    + n_layer * (12 * n_embd * n_embd + 13 * n_embd)
    + 2 * n_embd
)
print(f"Total parameters: {total_params:,}")
