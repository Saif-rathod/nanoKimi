# nanoKimi/export.py
import torch

def export_onnx(model, ckpt, out_path="out/model.onnx", example_input=None):
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    if example_input is None:
        import torch
        example_input = torch.randint(0, model.cfg.vocab_size, (1, 8), dtype=torch.long)
    torch.onnx.export(model, (example_input,), out_path, opset_version=14)
    print("Exported ONNX to", out_path)
