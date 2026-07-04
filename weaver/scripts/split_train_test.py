import argparse
import numpy as np
import awkward as ak


def main():
    parser = argparse.ArgumentParser(description='Split an extracted hidden-neuron parquet file into a '
                                      'trainval set (for weaver --data-train, which auto-splits train/val '
                                      'internally) and a held-out test set (untouched during training), '
                                      'stratified by _label_ so class proportions are preserved in both.')
    parser.add_argument('input_parquet')
    parser.add_argument('trainval_output')
    parser.add_argument('test_output')
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    arr = ak.from_parquet(args.input_parquet)
    n = len(arr)
    labels = ak.to_numpy(arr['_label_'])

    rng = np.random.default_rng(args.seed)
    test_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * args.test_frac))
        test_mask[idx[:n_test]] = True

    test_idx = np.where(test_mask)[0]
    trainval_idx = np.where(~test_mask)[0]

    # The source file is class-ordered (all of one class's events, then the next), inherited from
    # how Step 1 concatenated input files. weaver's internal train/val split (via --train-val-split)
    # is a positional slice of row order, NOT a shuffle -- so row order must be randomized here,
    # or train/val can end up with disjoint classes.
    rng.shuffle(trainval_idx)
    rng.shuffle(test_idx)

    ak.to_parquet(arr[trainval_idx], args.trainval_output, compression='LZ4', compression_level=4)
    ak.to_parquet(arr[test_idx], args.test_output, compression='LZ4', compression_level=4)

    print(f'trainval: {len(trainval_idx)} jets -> {args.trainval_output}')
    print(f'test: {len(test_idx)} jets -> {args.test_output}')


if __name__ == '__main__':
    main()
