import onnx

model = onnx.load("/home/testing/mobile_ocr/trion_repo/yolov11m/1/model.onnx")

print("=== MODEL INPUTS ===")
for inp in model.graph.input:
    print(f"Name: {inp.name}")
    print(f"Shape: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print()

print("=== MODEL OUTPUTS ===")
for out in model.graph.output:
    print(f"Name: {out.name}")
    print(f"Shape: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    print()