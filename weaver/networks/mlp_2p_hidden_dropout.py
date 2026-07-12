import torch
import torch.nn as nn

def layer(in_dim, out_dim, p=0.0):
    modules = [
        nn.Linear(in_dim, out_dim),
        # nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    ]
    # 14_hidden_dropout: p=0.0 (default, used for premlp) adds nothing; the main mlp hidden
    # layers pass p>0 to randomly zero a fraction of activations each forward pass (training only
    # -- disabled automatically in eval/test mode).
    if p > 0:
        modules.append(nn.Dropout(p=p))
    return nn.Sequential(*modules)

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
                 dropout_p=0.0, **kwargs):

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
        # 14_hidden_dropout: dropout applied only to the main mlp's hidden layers (not premlp,
        # not the final classification layer -- same rationale as excluding ReLU from the last
        # layer in 09).
        self.mlp = nn.Sequential(
            *[layer(in_dim, out_dim, p=dropout_p) for in_dim, out_dim in zip(channels[:-2], channels[1:-1])],
            nn.Linear(channels[-2], channels[-1]),
        )

    def forward(self, xp, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C) (no last dimension P as we set length = None)
        return self.mlp(torch.cat((self.premlp(xp), x), dim=1))


def get_model(data_config, **kwargs):
    prelayer_params = (32, 32)
    # 10_hidden_layers: first real hidden-layer capacity in the main mlp path (288 -> 128 -> 64 ->
    # num_classes), instead of the single linear readout used in every prior run (01-09).
    layer_params = (128, 64)
    # 14_hidden_dropout: regularize the hidden layers with dropout instead of weight decay
    # (11/12/13 showed weight decay only trades overfitting for underfitting, never beating the
    # 09_relu_fix baseline). weight_decay is back to 0 (not passed via --optimizer-option) so this
    # isolates dropout's effect alone.
    # 15_hidden_dropout_0p15: p=0.3 (14) already fully closed the train/val gap (eval-mode train
    # 0.6299 vs test 0.6292) with no headroom left -- tried a smaller value (0.15) to see if some
    # of the remaining gap to 09's baseline (0.6413) could be recovered without reopening the gap.
    # Result: p=0.15 was uniformly worse (test 0.6221) despite showing the same near-zero
    # train/test gap (0.6228 vs 0.6221) -- not overfitting, just a worse fit overall.
    # 16_hidden_dropout_0p5: trying a larger value to see whether p=0.3 sits at a real peak
    # (both directions worse) or whether this is mostly run-to-run noise.
    dropout_p = 0.5
    preinput_dims = len(data_config.input_dicts['basic'])
    input_dims = len(data_config.input_dicts['highlevel'])
    num_classes = len(data_config.label_value)
    model = MultiLayerPerceptron2Path(preinput_dims, input_dims, num_classes, prelayer_params=prelayer_params, layer_params=layer_params, dropout_p=dropout_p)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
    return model, model_info


def get_loss(data_config, class_weights=None, **kwargs):
    if class_weights is None:
        return torch.nn.CrossEntropyLoss()
    # train.py never calls `.to(dev)` on the loss function (only on the model), so a plain
    # `CrossEntropyLoss(weight=...)` would leave its weight tensor stuck on CPU while logits
    # run on GPU -- move it to the logits' device on every call instead.
    weight = torch.tensor(class_weights, dtype=torch.float32)

    def weighted_cross_entropy(input, target):
        return torch.nn.functional.cross_entropy(input, target, weight=weight.to(input.device))

    return weighted_cross_entropy