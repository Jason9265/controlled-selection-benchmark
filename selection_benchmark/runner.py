"""Dataset-dependent source training, receipt adaptation, and delayed holdout scoring."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _deps(deeplab_source: str, device: str):
    import sys
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    source = str(Path(deeplab_source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    import network
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    class Crops(Dataset):
        def __init__(self, rows, workspace: Path, augment: bool):
            self.rows, self.workspace, self.augment = list(rows), workspace, augment
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            image = Image.open(self.workspace / row["image"]).convert("RGB")
            mask = Image.open(self.workspace / row["mask"])
            if self.augment and random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            return x, torch.from_numpy(np.asarray(mask, dtype=np.int64))

    return torch, F, DataLoader, Crops, network


def _seed(torch, seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _metric(torch, model, loader, device: str):
    confusion = torch.zeros((6, 6), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            valid = masks != 255
            confusion += torch.bincount((masks[valid] * 6 + predictions[valid]).reshape(-1), minlength=36).reshape(6, 6)
    intersection = confusion.diag().double()
    union = confusion.sum(0).double() + confusion.sum(1).double() - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return {"miou_fixed": float(iou.mean()), "per_class_iou": [float(value) for value in iou], "valid_pixels": int(confusion.sum())}


def train_source(args):
    workspace, artifacts = Path(args.workspace).resolve(), Path(args.artifacts).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((workspace / "prepared" / "manifest.json").read_text())
    protocol = json.loads((artifacts / "isprs" / "protocol_frozen.json").read_text())
    spec = protocol["directions"][args.direction]
    train_groups, validation_groups = set(spec["source_train"]), set(spec["source_val"])
    if train_groups & validation_groups:
        raise RuntimeError("source split overlap")
    rows = manifest["rows"]
    train_rows = [row for row in rows if row["group"] in train_groups]
    validation_rows = [row for row in rows if row["group"] in validation_groups]
    torch, F, DataLoader, Crops, network = _deps(args.deeplab_source, args.device)
    _seed(torch, args.seed)
    model = network.modeling.deeplabv3plus_resnet101(num_classes=6, output_stride=16, pretrained_backbone=False).to(args.device)
    weights = Path(args.imagenet_resnet101).resolve()
    state = torch.load(weights, map_location="cpu", weights_only=False)
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key not in {"fc.weight", "fc.bias"}]
    if unexpected or any(not key.startswith("fc.") for key in missing):
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(Crops(train_rows, workspace, True), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, generator=generator)
    validation = DataLoader(Crops(validation_rows, workspace, False), batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    iterator, best_metric, best_state, history = iter(train), -1.0, None, []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            images, masks = next(iterator)
        except StopIteration:
            iterator = iter(train)
            images, masks = next(iterator)
        model.train()
        learning_rate = max(.01 * (1 - step / args.steps) ** .9, 1e-6)
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images.to(args.device)), masks.to(args.device), ignore_index=255)
        loss.backward(); optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metric = _metric(torch, model, validation, args.device)
            history.append({"step": step, "loss": float(loss.detach()), "lr": learning_rate, "source_validation": metric})
            if metric["miou_fixed"] > best_metric:
                best_metric = metric["miou_fixed"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output.mkdir(parents=True)
    checkpoint = output / "model_best_source_val.pt"
    torch.save(best_state, checkpoint)
    result = {"status": "complete", "direction": args.direction, "seed": args.seed, "steps": args.steps, "source_train_crops": len(train_rows), "source_validation_crops": len(validation_rows), "best_source_val_miou": best_metric, "history": history, "checkpoint_sha256": _sha(checkpoint), "target_labels_read": False, "target_holdout_labels_read": False, "deterministic_algorithms_requested": True, "deterministic_algorithms_enforced": False, "seconds": time.perf_counter() - start}
    (output / "source_train.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "best_source_val_miou": best_metric}))


def run_receipt(args):
    workspace, artifacts, output = Path(args.workspace).resolve(), Path(args.artifacts).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((workspace / "prepared" / "manifest.json").read_text())
    protocol = json.loads((artifacts / "isprs" / "protocol_frozen.json").read_text())
    config = json.loads((artifacts / "isprs" / "isprs_control_config.json").read_text())
    receipts = json.loads(Path(args.receipt_file).resolve().read_text())["receipts"]
    receipt = receipts[args.index]
    spec = protocol["directions"][receipt["direction"]]
    rows = manifest["rows"]
    selected = [row for row in rows if row["id"] in set(receipt["ids"])]
    source = [row for row in rows if row["group"] in set(spec["source_train"])]
    holdout = [row for row in rows if row["group"] in set(spec["target_holdout"])]
    if len(selected) != len(receipt["ids"]) or set(row["id"] for row in selected) & set(row["id"] for row in holdout):
        raise RuntimeError("receipt violates target-pool/holdout isolation")
    torch, F, _, Crops, network = _deps(args.deeplab_source, args.device)
    _seed(torch, receipt["model_seed"])
    model = network.modeling.deeplabv3plus_resnet101(num_classes=6, output_stride=16, pretrained_backbone=False).to(args.device)
    checkpoint = Path(args.source_checkpoint).resolve()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=.9, weight_decay=1e-4)
    import numpy as np
    rng = np.random.default_rng(receipt["model_seed"])
    source_data, selected_data = Crops(source, workspace, True), Crops(selected, workspace, True)
    for _ in range(config["adaptation_steps"]):
        source_indices = rng.integers(len(source_data), size=config["batch_size_source"])
        target_indices = rng.integers(len(selected_data), size=config["batch_size_target"])
        source_x, source_y = zip(*(source_data[int(index)] for index in source_indices))
        target_x, target_y = zip(*(selected_data[int(index)] for index in target_indices))
        images, labels = torch.stack([*source_x, *target_x]).to(args.device), torch.stack([*source_y, *target_y]).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels, ignore_index=255)
        loss.backward(); optimizer.step()
    output.mkdir(parents=True)
    model_path = output / "model.pt"
    torch.save(model.state_dict(), model_path)
    result = {"status": "complete", "receipt_index": args.index, "receipt": receipt, "steps": config["adaptation_steps"], "final_loss": float(loss.detach()), "model_sha256": _sha(model_path), "source_checkpoint_sha256": _sha(checkpoint), "target_holdout_labels_read_by_training": False, "target_pool_masks_read_for_selection": False, "deterministic_algorithms_requested": True, "deterministic_algorithms_enforced": False}
    (output / "adapt_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "final_loss": result["final_loss"]}))


def evaluate_receipt(args):
    workspace, artifacts, run_root = Path(args.workspace).resolve(), Path(args.artifacts).resolve(), Path(args.run_root).resolve()
    result_path = run_root / "blind_holdout.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite one-time holdout result {result_path}")
    completed = json.loads((run_root / "adapt_complete.json").read_text())
    receipt = completed["receipt"]
    manifest = json.loads((workspace / "prepared" / "manifest.json").read_text())
    protocol = json.loads((artifacts / "isprs" / "protocol_frozen.json").read_text())
    holdout_groups = set(protocol["directions"][receipt["direction"]]["target_holdout"])
    holdout_rows = [row for row in manifest["rows"] if row["group"] in holdout_groups]
    torch, _, DataLoader, Crops, network = _deps(args.deeplab_source, args.device)
    model = network.modeling.deeplabv3plus_resnet101(num_classes=6, output_stride=16, pretrained_backbone=False).to(args.device)
    model.load_state_dict(torch.load(run_root / "model.pt", map_location="cpu", weights_only=True))
    metric = _metric(torch, model, DataLoader(Crops(holdout_rows, workspace, False), batch_size=4, shuffle=False, num_workers=0), args.device)
    result = {"receipt_index": completed["receipt_index"], **metric, "holdout_crops": len(holdout_rows), "target_holdout_used_for_selection": False}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(result_path), "miou_fixed": metric["miou_fixed"]}))
