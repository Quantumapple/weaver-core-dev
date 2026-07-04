import argparse
import torch

# cpf is input collection index 0 (cpf, npf, sv order in our network config).
# Embed = BatchNorm1d(input_dim) -> [LayerNorm(input_dim), Linear(input_dim, dim0), GELU, ...]
# so only the first LayerNorm/Linear's input-side params depend on the 30-vs-24 cpf_features count.
KEYS_1D = [
    'input_embeds.0.input_bn.weight',
    'input_embeds.0.input_bn.bias',
    'input_embeds.0.input_bn.running_mean',
    'input_embeds.0.input_bn.running_var',
    'input_embeds.0.embed.0.weight',
    'input_embeds.0.embed.0.bias',
]
KEY_2D = 'input_embeds.0.embed.1.weight'
N_ORIG = 30
N_KEEP = 24


def main():
    parser = argparse.ArgumentParser(description='Slice the frozen GloParT checkpoint\'s cpf Embed layer '
                                      'from 30 to 24 input columns (dropping the 6 tracker hit-layer-count '
                                      'features we don\'t have in our NanoAOD).')
    parser.add_argument('input_ckpt')
    parser.add_argument('output_ckpt')
    args = parser.parse_args()

    state_dict = torch.load(args.input_ckpt, map_location='cpu')

    for key in KEYS_1D:
        orig = state_dict[key]
        assert orig.shape == (N_ORIG,), f'{key}: expected shape ({N_ORIG},), got {tuple(orig.shape)}'
        state_dict[key] = orig[:N_KEEP].clone()

    orig = state_dict[KEY_2D]
    assert orig.shape[1] == N_ORIG, f'{KEY_2D}: expected in_features={N_ORIG}, got {orig.shape[1]}'
    state_dict[KEY_2D] = orig[:, :N_KEEP].clone()

    torch.save(state_dict, args.output_ckpt)
    print(f'Wrote sliced checkpoint ({N_KEEP} cpf input columns, from {N_ORIG}) to {args.output_ckpt}')


if __name__ == '__main__':
    main()
