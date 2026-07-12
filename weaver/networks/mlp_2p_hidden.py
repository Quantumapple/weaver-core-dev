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
    # 10_hidden_layers: first real hidden-layer capacity in the main mlp path (288 -> 128 -> 64 ->
    # num_classes), instead of the single linear readout used in every prior run (01-09).
    # 17_smaller_hidden_layers: 14/15/16 (dropout on the 128,64 architecture) showed test accuracy
    # rising monotonically with dropout strength (up to p=0.5) with no sign of underfitting --
    # suggesting 128,64 has more capacity than this problem needs. Testing a much smaller
    # architecture (a quarter of the units: 32,16) directly, with dropout back to 0, to see if the
    # capacity reduction alone (no regularization needed) avoids the severe overfitting seen in 10.
    layer_params = (32, 16)
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