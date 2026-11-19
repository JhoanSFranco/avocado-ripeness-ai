import torch

from avocado_ai.ml.model import AvocadoRipenessNet, ModelMetadata, load_model, save_checkpoint


def build_model() -> AvocadoRipenessNet:
    torch.manual_seed(0)
    return AvocadoRipenessNet(num_classes=5, dropout=0.4).eval()


def test_forward_shapes():
    model = build_model()
    logits, regression = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 5)
    assert regression.shape == (2,)


def test_eval_forward_is_deterministic():
    model = build_model()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a, _ = model(images)
        b, _ = model(images)
    assert torch.equal(a, b)


def test_mc_dropout_outputs_and_keeps_eval_mode():
    model = build_model()
    out = model.predict_with_uncertainty(torch.randn(3, 3, 64, 64), num_samples=10)

    assert out.probs_mean.shape == (3, 5)
    assert torch.allclose(out.probs_mean.sum(dim=1), torch.ones(3), atol=1e-5)
    assert out.regression_mean.shape == (3,)
    assert (out.probs_var > 0).any(), "dropout must make the samples differ"
    assert not model.training, "MC sampling must not switch the model to train mode"


def test_backbone_freezing():
    model = AvocadoRipenessNet(trainable_backbone_layers=("layer4",))
    # features[7] is layer4, features[0] is conv1.
    assert all(p.requires_grad for p in model.features[7].parameters())
    assert not any(p.requires_grad for p in model.features[0].parameters())


def test_checkpoint_roundtrip(tmp_path):
    model = build_model()
    metadata = ModelMetadata(target_mean=6.2, target_std=6.4, metrics={"test_accuracy": 0.8})
    path = save_checkpoint(tmp_path / "model.pth", model, metadata, {"val_loss": [1.0, 0.5]})

    loaded, loaded_meta = load_model(path)  # uses weights_only=True internally

    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(images)[0], loaded(images)[0])
    assert loaded_meta.target_std == 6.4
    assert loaded_meta.metrics["test_accuracy"] == 0.8
    assert loaded_meta.denormalize_days(1.0) == 6.2 + 6.4
