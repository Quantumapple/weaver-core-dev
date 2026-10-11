import torch
import torch.nn as nn

# special_run_for_kappa_search: NOT part of the main numbered run_outputs/NN_name pipeline. Forked
# from mlp_2p_fixed.py rather than edited in place, so this experimental change never touches the
# network script the main run sequence depends on. Pairs with
# ak8_VJets_tagger_kappa_search.yaml / ak8_VJets_hidneuron_extraction_kappa_search.yaml.
#
# Instead of a fixed, precomputed jet-charge scalar (run 35's fj_charge_k0p0/k0p3/k0p5/k1p0), kappa
# itself is an nn.Parameter here: Q_kappa = (1/fj_pt^kappa) * sum_i(charge_i * pt_i^kappa) is computed
# from the per-candidate signed-pT (charge*pt) input inside forward(), so the same optimizer
# that trains the MLP head also fits kappa by gradient descent -- a continuous search, not a scan over
# a handful of fixed values.

def layer(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        # nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )


class LearnableJetCharge(nn.Module):
    r"""Q_kappa = (1/fj_pt^kappa) * sum_i(charge_i * pt_i^kappa), kappa a trainable scalar.

    kappa is parametrized as kappa_max * sigmoid(raw) rather than a bare nn.Parameter: without a
    bound, a bad gradient step could push kappa sharply negative/large, and pt_i^kappa (pt_i up to
    ~O(1000) GeV for candidates in high-pT jets) over/underflows fast outside a modest range. The
    literature values this project found (ATLAS/boosted-W/Z studies) all sit in [0, 1], so
    kappa_max=1.5 gives headroom either side without risking that blowup.
    """

    def __init__(self, init_kappa=0.5, kappa_max=1.5, eps=1e-3):
        super().__init__()
        init_kappa = min(max(init_kappa, 1e-3), kappa_max - 1e-3)
        init_raw = torch.logit(torch.tensor(init_kappa / kappa_max))
        self.raw = nn.Parameter(init_raw)
        self.kappa_max = kappa_max
        self.eps = eps

    @property
    def kappa(self):
        return torch.sigmoid(self.raw) * self.kappa_max

    def forward(self, cand_spt, fj_pt):
        # cand_spt: (N, 1, 90) signed pT = charge * pt, zero-padded. sign = charge (+-1), |.| = pT;
        # padding (0) has sign 0 so it contributes nothing -- no mask needed.
        s = cand_spt[:, 0, :]
        charge = torch.sign(s)
        pt = s.abs().clamp(min=self.eps)
        kappa = self.kappa
        weighted_sum = (charge * pt.pow(kappa)).sum(dim=1)
        return weighted_sum / fj_pt.clamp(min=self.eps).pow(kappa)


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

        self.jet_charge = LearnableJetCharge()
        # +1: the learned Q_kappa scalar, concatenated onto `basic` (fj_pt, fj_msoftdrop) below.
        prechannels = [preinput_dims + 1] + list(prelayer_params)
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

    def forward(self, xp, x, cand_spt):
        # xp: `basic` (N, C) -- [fj_pt, fj_msoftdrop]. fj_pt is column 0, matching
        # ak8_VJets_tagger_kappa_search.yaml's `basic.vars` order.
        # x: `highlevel` (N, C), the 256-dim frozen-backbone embedding.
        # cand_spt: (N, 1, 90) -- signed pT (charge*pt) per padded candidate slot, 0 for padding.
        q_kappa = self.jet_charge(cand_spt, xp[:, 0]).unsqueeze(1)
        xp_aug = torch.cat((xp, q_kappa), dim=1)
        return self.mlp(torch.cat((self.premlp(xp_aug), x), dim=1))


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
