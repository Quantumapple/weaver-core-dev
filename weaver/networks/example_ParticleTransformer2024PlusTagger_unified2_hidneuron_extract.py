import os
import numpy as np
import torch
import tqdm
from collections import defaultdict

from utils.logger import _logger
from utils.nn.tools import _concat
from utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer2024Plus.py'), 'ParT'
).ParticleTransformerTagger_ncoll

# Frozen checkpoint layout (train_GloParT_v3beta4.sh): num_cls_nodes=374, num_nodes=750
# (374 cls + 376 reg), embed_dims[-1]=256 hidden neurons.
N_CLS = 374
N_HIDDEN = 256


def get_model(data_config, **kwargs):
    # Hyperparameters pinned to train_GloParT_v3beta4.sh, which produced this checkpoint.
    # for_inference + export_params make the frozen model return one flat tensor:
    # [cls_softmax(374), reg_nodes(376), x_cls hidden embedding(256)] = 1006 columns.
    cfg = dict(
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
        num_classes=750,
        pair_input_type='pp',
        pair_input_dim=6,
        pair_extra_dim=0,
        use_pair_norm=True,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(256, 1024, 256),
        pair_embed_dims=(64, 64, 64),
        num_heads=16,
        num_layers=12,
        num_cls_layers=2,
        block_params={"scale_attn_mask": True, "scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
        cls_block_params={"scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
        fc_params=[(2048, 0.1)],
        activation='gelu',
        trim=True,
        for_inference=True,
        export_params={'concat_hid': True, 'num_cls': N_CLS, 'apply_softmax': True},
    )
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerTagger_ncoll(**cfg)
    model.num_cls_nodes = N_CLS

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    # predict-only: no training happens with this network config
    return None


def get_train_fn(data_config, **kwargs):
    def _no_train(*args, **kwargs):
        raise RuntimeError('This network config is predict-only (frozen-backbone hidden-neuron extraction)')
    return _no_train


def evaluate_predict(model, test_loader, dev, epoch, for_training=False, loss_func=None,
                      steps_per_epoch=None, tb_helper=None, **kwargs):
    model.eval()
    data_config = test_loader.dataset.config

    scores_cls, scores_reg = [], []
    labels = defaultdict(list)
    observers = defaultdict(list)
    num_batches = 0
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                model_output = model(*inputs)
                scores_cls.append(model_output[:, :N_CLS].float().cpu().numpy())
                scores_reg.append(model_output[:, N_CLS:].float().cpu().numpy())
                for k, v in y.items():
                    labels[k].append(v.cpu().numpy())
                for k, v in Z.items():
                    observers[k].append(v.cpu().numpy())
                num_batches += 1
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    scores_cls = np.concatenate(scores_cls)
    scores_reg = np.concatenate(scores_reg)
    labels = {k: _concat(v) for k, v in labels.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    if for_training:
        return 0.
    return 0., (scores_cls, scores_reg), labels, observers


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_predict


def get_save_fn(data_config, **kwargs):
    def save_fn(args, data_config, scores, labels, observers):
        scores_cls, scores_reg = scores
        output = {}

        # scores_reg = [regression nodes...] + [x_cls hidden embedding (last N_HIDDEN cols)]
        n_reg = scores_reg.shape[1] - N_HIDDEN
        hidden = scores_reg[:, n_reg:]
        for i in range(N_HIDDEN):
            output[f'fj_ParT_hidNeuron{i:03d}'] = hidden[:, i]

        for k, v in labels.items():
            if v.ndim == 1:
                output[k] = v
        for k, v in observers.items():
            if v.ndim == 1:
                output[k] = v

        lengths = {k: len(v) for k, v in output.items()}
        majority_len = max(set(lengths.values()), key=list(lengths.values()).count)
        mismatched = {k: n for k, n in lengths.items() if n != majority_len}
        if mismatched:
            _logger.warning('save_fn: length mismatch vs majority (%d): %s', majority_len, mismatched)

        return output
    return save_fn
