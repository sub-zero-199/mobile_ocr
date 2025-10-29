import os
import yaml

dataset_path = "/home/testing/mobile_ocr/dataset/lpds_v11yolov11"
yaml_path = os.path.join(dataset_path, "data.yaml")

print("🔍 Checking dataset structure...")
print(f"Dataset path: {dataset_path}")

# Check folders
folders_to_check = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'val/images', 'val/labels']
found_folders = {}

for folder in folders_to_check:
    full_path = os.path.join(dataset_path, folder)
    exists = os.path.exists(full_path)
    found_folders[folder] = exists
    status = "✅" if exists else "❌"
    print(f"{status} {folder}: {exists}")

# Check data.yaml
if os.path.exists(yaml_path):
    print(f"\n📄 data.yaml content:")
    with open(yaml_path, 'r') as f:
        content = yaml.safe_load(f)
        print(yaml.dump(content, default_flow_style=False))
else:
    print(f"❌ data.yaml not found at {yaml_path}")

# Suggest fix
print("\n💡 Suggestions:")
if found_folders.get('val/images') and not found_folders.get('valid/images'):
    print("   - Rename 'val' folder to 'valid' OR update data.yaml to use 'val'")
elif not found_folders.get('valid/images') and not found_folders.get('val/images'):
    print("   - Validation folder is completely missing!")
    print("   - Check if dataset was extracted correctly")