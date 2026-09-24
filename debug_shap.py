
import torch
import torch.nn as nn
import shap
class M(nn.Module):
    def forward(self, x, y):
        return (x.sum(dim=-1) + y.sum(dim=-1)).unsqueeze(-1)
x = torch.ones(10, 2)
y = torch.ones(10, 3)
explainer = shap.GradientExplainer(M(), [x, y])
v = explainer.shap_values([x[:1], y[:1]])
print(type(v))
if isinstance(v, list):
    print('List len:', len(v))
    print('Types in list:', [type(item) for item in v])
    if isinstance(v[0], list):
        print('List of Lists! inner len:', len(v[0]))
        print('Types inside inner list:', [type(item) for item in v[0]])
else:
    print('Shape:', v.shape)
