# generate_dataset.py
from src.data.data_generator import SmartGridDataGenerator

print('='*60)
print('GENERATING DATASET')
print('='*60)

generator = SmartGridDataGenerator(
    n_agents=5,        # 20 households
    n_days=365,         # 1 year of data
    seed=42
)

data = generator.generate_full_dataset()
generator.save_dataset(data, 'data/processed/dataset_20agents_365days.npz')
generator.visualize_dataset(data, days_to_plot=7)

print('\n✓ Dataset generation complete!')