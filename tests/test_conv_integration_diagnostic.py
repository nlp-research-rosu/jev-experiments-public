"""Independent contracts for the bounded convolution integration diagnostic."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'reports/post-pause-audit-v1/compare_convolution.py'
spec = importlib.util.spec_from_file_location('conv_diagnostic', PATH)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_group_probabilities_have_one_binary_logit_and_whole_softmax():
    groups = [('noul', (0,)), ('choice', (1, 2, 3)), ('score', (4, 5))]
    probs, labels = diag.group_probabilities(torch.tensor([1.0986123, 1., 2., 3., 0., 0.]), groups)
    assert probs[:2].tolist() == pytest.approx([.25, .75])
    assert probs[2:5].sum().item() == pytest.approx(1.)
    assert probs[5:].tolist() == pytest.approx([.5, .5])
    assert labels == [1, 2, 0]


def test_tensor_comparison_checks_all_names_and_rejects_nonfinite():
    a = {'a': torch.tensor([3., 4.]), 'b': torch.tensor([0.])}
    b = {'b': torch.tensor([0.]), 'a': torch.tensor([3., 4.1])}
    result = diag.tensor_difference(b, a)
    assert result['max_absolute'] == pytest.approx(.1, abs=1e-6)
    assert result['relative_l2'] == pytest.approx(.02, abs=1e-6)
    assert set(result['per_tensor']) == {'a', 'b'}
    with pytest.raises(ValueError, match='names'):
        diag.tensor_difference({'x': torch.tensor([1.])}, a)
    with pytest.raises(ValueError, match='nonfinite'):
        diag.tensor_difference({'a': torch.tensor([float('nan'), 4.]), 'b': torch.tensor([0.])}, a)


def test_comparison_cannot_pass_changed_decision_or_resume_update():
    metric = {'max_absolute': 0., 'relative_l2': 0.}
    base = {'logits': metric, 'probabilities': metric, 'gradients': metric,
            'parameters': metric, 'loss_difference': 0., 'max_component_loss_difference': 0.,
            'class_changes': 0}
    assert diag.check_gates(base, exact=False) == []
    assert diag.check_gates({**base, 'class_changes': 1}, exact=False)
    assert diag.check_gates({**base, 'parameters': {'max_absolute': 2e-7}}, exact=True)
    assert diag.check_gates({**base, 'gradients': {'max_absolute': .5, 'relative_l2': .03}}, exact=False)
