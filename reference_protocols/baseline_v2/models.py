"""Frozen architecture interfaces for ResNet18 U-Net and ImageNet MiT-B0 SegFormer.

Input is RGB float Bx3xHxW in [0,1]; ImageNet normalization is internal.
No target labels, remote model loading, automatic downloads, or random fallback.
SegFormer uses the pinned Hugging Face reference implementation, the NVIDIA
ImageNet classification encoder, and a freshly initialized semantic decoder.
"""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
MIT_DIRECTORY = ROOT / 'weights/mit-b0'
MIT_REVISION = '80983a413c30d36a39c20203974ae7807835e2b4'
TRANSFORMERS_VERSION = '4.57.6'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _unet_class():
    path = ROOT.parent / 'baseline/model.py'
    spec = importlib.util.spec_from_file_location('_baseline_v1_unet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNet18UNet


def _transformers():
    vendor = str(ROOT / 'vendor')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f'Expected vendored transformers {TRANSFORMERS_VERSION}, got {transformers.__version__}')
    from transformers import SegformerConfig
    from transformers.models.segformer.modeling_segformer import SegformerForSemanticSegmentation
    return SegformerConfig, SegformerForSemanticSegmentation


class SegFormerB0(nn.Module):
    """Official MiT-B0 encoder with the reference all-MLP SegFormer decode head.

    Encoder LayerNorm, stochastic depth .1, decoder BatchNorm and dropout .1
    follow the pinned reference defaults. Decoder BN is trainable, unlike the
    reused U-Net's fixed encoder BN. Upsampling is reference bilinear with
    align_corners=False. Strict CUDA determinism is checked by the smoke report.
    """
    def __init__(self, classes=8, pretrained=True):
        super().__init__()
        if not isinstance(classes, int) or isinstance(classes, bool) or classes < 1:
            raise ValueError('classes must be a positive integer')
        if not isinstance(pretrained, bool):
            raise TypeError('pretrained must be a bool')
        Config, Reference = _transformers()
        if pretrained:
            receipt = json.loads((ROOT / 'sources/segformer_checkpoint_receipt.json').read_text())
            if receipt['revision'] != MIT_REVISION:
                raise ValueError('MiT-B0 checkpoint revision mismatch')
            for name in ('config.json', 'pytorch_model.bin'):
                if _sha(MIT_DIRECTORY / name) != receipt['files'][name]['sha256']:
                    raise ValueError(f'MiT-B0 checksum mismatch: {name}')
            config = Config.from_dict(json.loads((MIT_DIRECTORY / 'config.json').read_text()))
        else:
            config = Config()
        config.num_labels = classes
        config.id2label = {i: str(i) for i in range(classes)}
        config.label2id = {str(i): i for i in range(classes)}
        config.reshape_last_stage = True
        reference = Reference(config)
        self.encoder = reference.segformer.encoder
        self.decode_head = reference.decode_head
        self.config = config
        if pretrained:
            state = torch.load(MIT_DIRECTORY / 'pytorch_model.bin', map_location='cpu', weights_only=True)
            prefix = 'segformer.encoder.'
            encoder_state = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
            unexpected = set(state) - {prefix + k for k in encoder_state} - {'classifier.weight', 'classifier.bias'}
            if unexpected or not encoder_state:
                raise ValueError(f'Unexpected ImageNet checkpoint keys: {sorted(unexpected)}')
            self.encoder.load_state_dict(encoder_state, strict=True)
            self.pretrained_metadata = {
                'pretrained': True, 'dataset': 'ImageNet-1K', 'repository': 'nvidia/mit-b0',
                'revision': MIT_REVISION, 'checkpoint_sha256': receipt['files']['pytorch_model.bin']['sha256'],
                'url': receipt['files']['pytorch_model.bin']['url'], 'head': 'fresh semantic decoder; classification head discarded',
                'encoder_load': 'strict, all keys matched', 'implementation': f'transformers {TRANSFORMERS_VERSION}',
                'license': receipt['license'],
            }
        else:
            self.pretrained_metadata = {'pretrained': False, 'head': 'fresh semantic decoder',
                                        'implementation': f'transformers {TRANSFORMERS_VERSION}'}
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32:
            raise ValueError('Expected RGB Bx3xHxW with H,W >=32')
        if not images.is_floating_point():
            raise TypeError('Expected floating RGB values in [0,1]')
        normalized = (images - self.image_mean) / self.image_std
        output = self.encoder(normalized, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states
        logits = self.decode_head(hidden)
        logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return logits, hidden[-1].mean(dim=(2, 3))


def build_model(arch, classes, pretrained=True):
    """Return model with .encoder, .pretrained_metadata and (logits, embedding)."""
    if arch in ('unet', 'resnet18_unet'):
        return _unet_class()(num_classes=classes, pretrained=pretrained,
                            weights_path=ROOT.parent / 'baseline/weights/resnet18-f37072fd.pth' if pretrained else None)
    if arch in ('segformer', 'segformer_b0', 'mit_b0'):
        return SegFormerB0(classes=classes, pretrained=pretrained)
    raise ValueError(f'Unknown architecture {arch!r}; use resnet18_unet or segformer_b0')
