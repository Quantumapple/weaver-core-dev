import os
import torch
import torch.nn as nn

from utils.logger import _logger
from utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer2024Plus.py'), 'ParT'
).ParticleTransformerTagger_ncoll

# Same checkpoint used for every frozen-embedding extraction in this project (GloParT v3beta4).
CHECKPOINT = 'dataset/ak8_MD_inclv10beta4_ul_manual.ddp4-bs640-lr1p2e-3.nepoch100.farm221/net_best_epoch_state.pt'


class VTaggerFinetune(nn.Module):
    """GloParT v3beta4 backbone (`ParticleTransformerTagger_ncoll`) + a linear readout head
    (same architecture as `mlp_2p_fixed.py`'s head, so results are comparable to 18/25/26/27).

    Backbone is initialized from `CHECKPOINT` and frozen except the last `num_unfrozen_blocks`
    of `part.blocks` (the regular transformer blocks). `part.cls_blocks` (the class-attention
    aggregator that produces x_cls), `part.norm`, and `part.cls_token` are always left trainable --
    they're the small aggregation step right before the head, not the token-mixing body, and
    freezing them would leave the head with no way to reweight which particle-level features it
    reads from a backbone that's otherwise fully frozen.
    """

    def __init__(self, num_classes, num_unfrozen_blocks=1, **backbone_kwargs):
        super().__init__()
        embed_dim = backbone_kwargs['embed_dims'][-1]
        backbone_kwargs['num_classes'] = None
        backbone_kwargs['fc_params'] = None  # no fc inside the backbone -> main(*inputs) returns x_cls directly
        self.main = ParticleTransformerTagger_ncoll(**backbone_kwargs)
        self.head = nn.Linear(embed_dim, num_classes)

        state = torch.load(CHECKPOINT, map_location='cpu')
        missing, unexpected = self.main.load_state_dict(state, strict=False)
        _logger.info('Loaded backbone weights from %s\n  Missing: %s\n  Unexpected: %s',
                      CHECKPOINT, missing, unexpected)

        num_layers = len(self.main.part.blocks)
        unfreeze_from = num_layers - num_unfrozen_blocks
        for name, param in self.main.named_parameters():
            if name.startswith('part.blocks.'):
                block_idx = int(name.split('.')[2])
                if block_idx >= unfreeze_from:
                    continue  # one of the last num_unfrozen_blocks -- stays trainable
            elif name.startswith('part.cls_blocks.') or name.startswith('part.norm') or name == 'part.cls_token':
                continue  # aggregator -- always trainable, see class docstring
            param.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        _logger.info('VTaggerFinetune: %d / %d params trainable (num_unfrozen_blocks=%d)',
                      n_trainable, n_total, num_unfrozen_blocks)

    def forward(self, *inputs):
        x_cls = self.main(*inputs)
        return self.head(x_cls)


def get_model(data_config, **kwargs):
    cfg = dict(
        num_classes=len(data_config.label_value),
        num_unfrozen_blocks=1,
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
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
        block_params={'scale_attn_mask': True, 'scale_attn': False, 'scale_fc': False,
                      'scale_heads': False, 'scale_resids': False, 'activation': 'swiglu'},
        cls_block_params={'scale_attn': False, 'scale_fc': False, 'scale_heads': False,
                           'scale_resids': False, 'activation': 'swiglu'},
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = VTaggerFinetune(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
                          **{'softmax': {0: 'N'}}},
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
