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
    dropout_p : float
        Dropout probability applied directly to the concatenated (premlp + highlevel) feature
        vector, right before the single linear readout. Unlike `mlp_2p_hidden_dropout.py` (which
        places dropout between hidden layers), this architecture has no hidden layers at all --
        34_input_dropout: same reference architecture as 30_reference_model (`mlp_2p_fixed.py`,
        a single Linear(288, num_classes) readout, no added capacity), with dropout applied to the
        288-dim input itself instead. This randomly zeroes a fraction of the frozen GloParT
        embedding dims (and premlp output dims) on every batch during training, which can improve
        generalization even without the kind of severe overfitting that motivated dropout in
        14/15/16 -- 30 itself showed only a small train/val gap (0.023), so this isn't fixing
        overfitting so much as testing whether input-level regularization helps at all.
    """

    def __init__(self, preinput_dims, input_dims, num_classes,
                 prelayer_params=(32, 32), dropout_p=0.0, **kwargs):

        super(MultiLayerPerceptron2Path, self).__init__(**kwargs)

        prechannels = [preinput_dims] + list(prelayer_params)
        self.premlp = nn.Sequential(
            *[layer(in_dim, out_dim) for in_dim, out_dim in zip(prechannels[:-1], prechannels[1:])]
        )
        concat_dim = input_dims + prechannels[-1]
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        # Bare Linear, no ReLU -- same rationale as 09_relu_fix: CrossEntropyLoss expects
        # unrestricted logits.
        self.final = nn.Linear(concat_dim, num_classes)

    def forward(self, xp, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C) (no last dimension P as we set length = None)
        feat = torch.cat((self.premlp(xp), x), dim=1)
        feat = self.dropout(feat)
        return self.final(feat)


def get_model(data_config, **kwargs):
    prelayer_params = (32, 32)
    # 34_input_dropout: starting at 0.3, the same first value used in 14, before sweeping further.
    dropout_p = 0.3
    preinput_dims = len(data_config.input_dicts['basic'])
    input_dims = len(data_config.input_dicts['highlevel'])
    num_classes = len(data_config.label_value)
    model = MultiLayerPerceptron2Path(preinput_dims, input_dims, num_classes, prelayer_params=prelayer_params, dropout_p=dropout_p)

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
