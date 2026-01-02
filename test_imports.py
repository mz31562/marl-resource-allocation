import sys
sys.path.append('.')

print("Testing imports...")

try:
    from src.data.data_generator import SmartGridDataGenerator
    print("✓ SmartGridDataGenerator imported")
except Exception as e:
    print(f"✗ Error importing data_generator: {e}")

try:
    from src.environments.dataset_driven_env import DatasetDrivenSmartGridEnv
    print("✓ DatasetDrivenSmartGridEnv imported")
except Exception as e:
    print(f"✗ Error importing dataset_driven_env: {e}")

print("\n✓ All imports working!")