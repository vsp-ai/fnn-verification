"""Rule-based Representation Learner (RRL, Wang et al. 2021) restricted to what the
FNN policies use: binary input atoms, logical layers of conjunctions and
disjunctions with novel logical activation functions (NLAF), and a linear output
layer, trained with gradient grafting."""
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLD = 0.5


class GradGraft(torch.autograd.Function):
    """Gradient grafting: forward the discrete output, backpropagate the continuous one."""
    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


def binarize(W):
    return torch.where(W - THRESHOLD > 0, torch.ones_like(W), torch.zeros_like(W))


def negate_atom_name(name):
    """'x eq 3' -> 'x neq 3', so negated atoms are plain atoms for the verifier."""
    negated = {'eq': 'neq', 'neq': 'eq', 'lt': 'geq', 'leq': 'gt', 'gt': 'leq', 'geq': 'lt'}
    parts = str(name).rsplit(' ', 2)
    if len(parts) == 3 and parts[1] in negated:
        return f'{parts[0]} {negated[parts[1]]} {parts[2]}'
    return '~' + str(name)


class InputLayer(nn.Module):
    """Binary input atoms, with their negations if use_not."""

    def __init__(self, input_dim, use_not):
        super().__init__()
        self.use_not = use_not
        self.output_dim = input_dim * 2 if use_not else input_dim
        self.dim2id = {i: i for i in range(self.output_dim)}

    def forward(self, x):
        return torch.cat((x, 1 - x), dim=1) if self.use_not else x

    def atom_names(self, feature_names):
        names = list(feature_names)
        return names + [negate_atom_name(n) for n in names] if self.use_not else names


class ConjunctionLayer(nn.Module):
    def __init__(self, n, input_dim, use_not, alpha, beta, gamma):
        super().__init__()
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, n))
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, x):
        return GradGraft.apply(self.binarized_forward(x), self.continuous_forward(x))

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        x = 1. - x
        xl = (1. - 1. / (1. - (x * self.alpha) ** self.beta))
        wl = (1. - 1. / (1. - (self.W * self.alpha) ** self.beta))
        return 1. / (1. + xl @ wl) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        res = (1 - x) @ binarize(self.W)
        return torch.where(res > 0, torch.zeros_like(res), torch.ones_like(res))


class DisjunctionLayer(ConjunctionLayer):
    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        xl = (1. - 1. / (1. - (x * self.alpha) ** self.beta))
        wl = (1. - 1. / (1. - (self.W * self.alpha) ** self.beta))
        return 1. - 1. / (1. + xl @ wl) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        res = x @ binarize(self.W)
        return torch.where(res > 0, torch.ones_like(res), torch.zeros_like(res))


def extract_rules(prev_layer, layer, node_activation_cnt, forward_tot, pos_shift):
    """Rules of one sublayer as sorted tuples of (k, rule id) inputs, k = -1 for a
    rule of the previous layer and k = 1 for its negation; dead or constant nodes
    and duplicate rules are dropped. Returns (node -> rule id, rule list)."""
    dim2id = defaultdict(lambda: -1)
    rules, rule_list = {}, []
    Wb = (layer.W.t() > THRESHOLD).type(torch.int).detach().cpu().numpy()
    prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
    for ri, row in enumerate(Wb):
        cnt = node_activation_cnt[ri + pos_shift]
        if cnt == 0 or cnt == forward_tot:
            dim2id[ri + pos_shift] = -1
            continue
        rule = {}
        for i, w in enumerate(row):
            use_not_mul = 1
            if layer.use_not:
                if i >= layer.input_dim // 2:
                    use_not_mul = -1
                i = i % (layer.input_dim // 2)
            if w > 0 and prev_dim2id[i][1] != -1:
                rid = prev_dim2id[i]
                rule[(rid[0] * use_not_mul, rid[1])] = 1
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = len(rule_list)
            rule_list.append(rule)
        dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list


class LogicalLayer(nn.Module):
    """n conjunctions followed by n disjunctions."""

    def __init__(self, n, input_dim, use_not, alpha, beta, gamma):
        super().__init__()
        self.output_dim = 2 * n
        self.con_layer = ConjunctionLayer(n, input_dim, use_not, alpha, beta, gamma)
        self.dis_layer = DisjunctionLayer(n, input_dim, use_not, alpha, beta, gamma)
        self.sublayers = [self.con_layer, self.dis_layer]

    def forward(self, x):
        return torch.cat([l(x) for l in self.sublayers], dim=1)

    def binarized_forward(self, x):
        return torch.cat([l.binarized_forward(x) for l in self.sublayers], dim=1)

    def clip(self):
        for l in self.sublayers:
            l.W.data.clamp_(0.0, 1.0)

    def get_rules(self, prev_layer):
        """Sets dim2id (node -> rule id over both sublayers) and rule_list (one list per sublayer)."""
        dim2id, rule_list, pos_shift, id_shift = {}, [], 0, 0
        for l in self.sublayers:
            sub_dim2id, sub_rules = extract_rules(prev_layer, l, self.node_activation_cnt, self.forward_tot, pos_shift)
            dim2id.update({k: (-1 if v == -1 else v + id_shift) for k, v in sub_dim2id.items()})
            rule_list.append(sub_rules)
            pos_shift += l.W.shape[1]
            id_shift += len(sub_rules)
        self.dim2id = defaultdict(lambda: -1, dim2id)
        self.rule_list = tuple(rule_list)


class LRLayer(nn.Module):
    """Linear output layer over the rules of the last logical layer."""

    def __init__(self, n, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n)

    def forward(self, x):
        return self.fc1(x)

    def get_rule2weights(self, prev_layer):
        """Bias with the weights of always-true rules folded in (self.bl), and the
        summed weights per rule, sorted by largest absolute weight (self.rule2weights)."""
        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot)
        prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        Wl, bl = list(self.fc1.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()
        marked = defaultdict(lambda: defaultdict(float))
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = prev_dim2id[i]
                if rid[1] == -1:
                    continue
                marked[rid][label_id] += w
        self.rule2weights = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)


class Net(nn.Module):
    def __init__(self, n_atoms, widths, n_outputs, use_not, alpha, beta, gamma, temperature):
        super().__init__()
        self.layer_list = nn.ModuleList([InputLayer(n_atoms, use_not)])
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))  # trainable softmax temperature (ce)
        for n in widths:
            self.layer_list.append(LogicalLayer(n, self.layer_list[-1].output_dim, use_not, alpha, beta, gamma))
        self.layer_list.append(LRLayer(n_outputs, self.layer_list[-1].output_dim))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

    @torch.no_grad()
    def count_activations(self, loader, device):
        """Number of samples on which each node of the logical layers is true."""
        for layer in self.layer_list[1:-1]:
            layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=device)
            layer.forward_tot = 0
        for x, _ in loader:
            x = self.layer_list[0](x.to(device))
            for layer in self.layer_list[1:-1]:
                x = layer.binarized_forward(x)
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]


def kl_loss(y_pred, y_true, temperature):
    """KL(softmax(teacher / T) || softmax(student)), temperature on the teacher scores only."""
    log_teacher = F.log_softmax(y_true.detach() / temperature, dim=-1)
    return F.kl_div(F.log_softmax(y_pred, dim=-1), log_teacher, reduction="batchmean", log_target=True)


def train(net, train_loader, valid_loader, device, loss, epochs, lr, lr_decay_rate, lr_decay_epoch, kl_temp):
    """Adam with step learning-rate decay; logical weights are clipped to [0, 1] after every step."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epo in range(epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (lr_decay_rate ** (epo // lr_decay_epoch))
        loss_sum, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_bar = net.forward(X)
            if loss == 'ce':
                batch_loss = F.cross_entropy(y_bar / torch.exp(net.t), torch.argmax(y, dim=1))
            elif loss == 'mse':
                batch_loss = ((y_bar - y) ** 2).mean()
            else:
                batch_loss = kl_loss(y_bar, y, kl_temp)
            loss_sum += batch_loss.item() * X.shape[0]
            n += X.shape[0]
            batch_loss.backward()
            optimizer.step()
            for layer in net.layer_list[1:-1]:
                layer.clip()
        print(f"epoch {epo + 1}/{epochs}  loss {loss_sum / n:.6f}  "
              f"valid fidelity {fidelity(net, valid_loader, device):.4f}", flush=True)


@torch.inference_mode()
def fidelity(net, loader, device):
    """Fraction of samples on which the network picks the teacher's action (argmax)."""
    correct = total = 0
    for X, y in loader:
        out = net.forward(X.to(device))
        correct += int((torch.argmax(out, dim=1).cpu() == torch.argmax(y, dim=1)).sum())
        total += y.shape[0]
    return correct / total
