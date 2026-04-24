from src.nn.data import DataGenerator


def generate_data():
    generator = DataGenerator(seed=42)
    generator.generateAllDatasets()


if __name__ == "__main__":
    generate_data()
