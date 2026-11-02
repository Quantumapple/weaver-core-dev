# 37_focal_loss: forked from mlp_2p_fixed.py. Architecture identical; only get_loss differs
# (focal loss instead of plain cross-entropy), so the loss is the single change vs. run 36.

import torch
import torch.nn as nn

def layer(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        # nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )

class MultiLayerPerceptron2Path(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, preinput_dims, input_dims, num_classes,
                 prelayer_params=(32, 32), layer_params=(1024, 256, 256),
                 **kwargs):
                
        super(MultiLayerPerceptron2Path, self).__init__(**kwargs)

        prechannels = [preinput_dims] + list(prelayer_params)
        self.premlp = nn.Sequential(
            *[layer(in_dim, out_dim) for in_dim, out_dim in zip(prechannels[:-1], prechannels[1:])]
        )
        channels = [input_dims + prechannels[-1]] + list(layer_params) + [num_classes]
        # 09_relu_fix: mlp_2p.py built every layer (including the final classification layer) via
        # `layer()`, which always appends a ReLU -- clipping the final class scores to >=0 right
        # before CrossEntropyLoss, which expects unrestricted logits. Keep ReLU between hidden
        # layers (standard), but leave the final layer a bare Linear.
        self.mlp = nn.Sequential(
            *[layer(in_dim, out_dim) for in_dim, out_dim in zip(channels[:-2], channels[1:-1])],
            nn.Linear(channels[-2], channels[-1]),
        )

    def forward(self, xp, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C) (no last dimension P as we set length = None)
        return self.mlp(torch.cat((self.premlp(xp), x), dim=1))


def get_model(data_config, **kwargs):
    prelayer_params = (32, 32)
    layer_params = ()
    preinput_dims = len(data_config.input_dicts['basic'])
    input_dims = len(data_config.input_dicts['highlevel'])
    num_classes = len(data_config.label_value)
    model = MultiLayerPerceptron2Path(preinput_dims, input_dims, num_classes, prelayer_params=prelayer_params, layer_params=layer_params)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
    return model, model_info


def get_loss(data_config, gamma=2.0, class_weights=None, **kwargs):
    r"""Multi-class focal loss (Lin et al. 2017): FL = (1 - p_t)^gamma * CE, mean-reduced.

    gamma=0 reduces exactly to cross-entropy. Set via `--network-option gamma <value>`.
    `class_weights` (optional, `--network-option class_weights "[...]"`) is the per-class alpha.
    """
    weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)

    def focal_loss(input, target):
        # train.py never calls `.to(dev)` on the loss function, so move the weight per call.
        w = None if weight is None else weight.to(input.device)
        ce = torch.nn.functional.cross_entropy(input, target, weight=w, reduction='none')
        # p_t = exp(-unweighted CE); recompute unweighted so alpha does not leak into (1 - p_t).
        logp_t = -torch.nn.functional.cross_entropy(input, target, reduction='none')
        return ((1 - logp_t.exp()).pow(gamma) * ce).mean()

    return focal_loss
