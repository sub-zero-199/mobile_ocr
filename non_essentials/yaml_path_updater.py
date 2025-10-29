import yaml

yaml_path = "/home/testing/mobile_ocr/dataset/s3_196_dataset/data.yaml"
base_path = "/home/testing/mobile_ocr/dataset/s3_196_dataset"

# Read current config
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

print("🔍 Current configuration:")
print(f"  train: {config.get('train')}")
print(f"  val: {config.get('val')}")
print(f"  test: {config.get('test')}")

# Check if changes are needed
needs_fix = (
    config.get('train') != f'{base_path}/train/images' or
    config.get('val') != f'{base_path}/valid/images' or
    config.get('test') != f'{base_path}/test/images'
)

if needs_fix:
    print("\n⚠️  Changes needed! Applying fixes...")
    
    # Apply absolute paths
    config['train'] = f'{base_path}/train/images'
    config['val'] = f'{base_path}/valid/images'
    config['test'] = f'{base_path}/test/images'
    
    # Write back (preserving structure, no 'path' key needed)
    with open(yaml_path, 'w') as f:
        f.write(f"train: {config['train']}\n")
        f.write(f"val: {config['val']}\n")
        f.write(f"test: {config['test']}\n\n")
        f.write(f"nc: {config['nc']}\n")
        f.write(f"names: {config['names']}\n\n")
        f.write("roboflow:\n")
        for key, value in config['roboflow'].items():
            f.write(f"  {key}: {value}\n")
    
    print("✅ Fixed with absolute paths!")
else:
    print("\n✅ Already using absolute paths - no changes needed!")

print("\n📄 Updated configuration:")
print(f"  train: {config['train']}")
print(f"  val: {config['val']}")
print(f"  test: {config['test']}")