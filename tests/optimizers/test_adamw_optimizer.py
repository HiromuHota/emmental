import logging
import shutil

import pytest
import torch
import torch.nn as nn
from packaging import version

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.2.0"),
    reason="at least torch-1.2.0 required",
)
def test_adamw_optimizer(caplog):
    """Unit test of AdamW optimizer"""

    caplog.set_level(logging.INFO)

    optimizer = "adamw"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default AdamW setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "amsgrad": False,
        "weight_decay": 0,
    }

    # Test new AdamW setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {
                    "betas": (0.9, 0.99),
                    "eps": 1e-05,
                    "amsgrad": True,
                },
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.02,
        "betas": (0.9, 0.99),
        "eps": 1e-05,
        "amsgrad": True,
        "weight_decay": 0.05,
    }

    shutil.rmtree(dirpath)
