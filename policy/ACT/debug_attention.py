import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from act_policy import ACTPolicy

# English
args_override = {
    "lr": 1e-5,
    "chunk_size": 50,
    "num_queries": 50,
    "kl_weight": 10.0,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone_type": "dinov2",
    "backbone": "vit_base_patch14_dinov2",
    "backbone_pretrained": False,
    "backbone_image_size": "224,224",
    "use_distillation": True,
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    "device": "cpu"
}

print("Initializing policy...")
policy = ACTPolicy(args_override)

# English
timm_body = policy.model.backbones[0][0].body
last_block = timm_body.blocks[-1]
attn_layer = last_block.attn

print("\n=== Attention Layer Structure ===")
print(f"Attention layer type: {type(attn_layer)}")
print(f"\nAttributes of attention layer:")
for attr in dir(attn_layer):
    if not attr.startswith('_'):
        val = getattr(attn_layer, attr)
        if not callable(val):
            print(f"  {attr}: {type(val)}")

# English qkv English
if hasattr(attn_layer, 'qkv'):
    print(f"\n=== QKV Layer ===")
    print(f"QKV layer type: {type(attn_layer.qkv)}")
    if hasattr(attn_layer.qkv, 'weight'):
        print(f"QKV weight shape: {attn_layer.qkv.weight.shape}")

# English
print("\n=== Testing Forward Pass ===")
test_input = torch.randn(1, 3, 224, 224)

# English hook English
attn_outputs = []
def test_hook(module, input, output):
    print(f"\n  Hook called on {module.__class__.__name__}")
    if isinstance(output, tuple):
        print(f"  Output is tuple with {len(output)} elements")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"    Output[{i}] shape: {o.shape}")
    else:
        if isinstance(output, torch.Tensor):
            print(f"  Output shape: {output.shape}")
    attn_outputs.append(output)

handle = attn_layer.register_forward_hook(test_hook)

try:
    with torch.no_grad():
        prepared = policy.model.backbones[0][0]._prepare_input(test_input)
        features = timm_body.forward_features(prepared)
        print(f"\n  Forward pass successful!")
        print(f"  Features shape: {features.shape}")
except Exception as e:
    print(f"\n  Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

handle.remove()

print("\n=== Done ===")
