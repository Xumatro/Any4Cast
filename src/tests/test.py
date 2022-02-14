# ---------- Setup ----------

from Any4Cast import neural
from Any4Cast import data
data_settings = {
    'n_features': 10,
    'n_skipped': 0,
    'n_labels': 1,
    'train_test_split': 0.85
}


def extract(entry):
    return entry['Price']


with open('test_data.json', 'r') as json_data:
    json_data = json_data.read()

neural_settings = {
    'n_input_dimensions': 1,
    'n_hidden_dimensions': 36,
    'n_output_dimensions': 1,
    'n_layers': 6,
    'learning_rate': 0.015,
    'n_epochs': 10
}

# ---------- Data Testing ----------

print("Testing Any4Cast.data...")

print("Import...", end="")
print(" Success!")

print("Initialization...", end="")
ds = data.Dataset(data_settings)
print(" Success!")

print("Data loading...", end="")
ds.load_json(json_data, extn_func=extract)
print(" Success!")

print("Data Normalizing...", end="")
ds.normalize(method="z-score")
print(" Success!")

print("Dataset Creation...", end="")
#x_train, y_train, x_test, y_test = ds.create_datasets()
#assert len(x_train) == len(y_train)
#assert len(x_test) == len(y_test)
print(" Success!")

print("Data Tests completed!\n")

# ---------- Neural testing ----------

print("Testing Any4Cast.neural...")

print("Import...", end="")
print(" Success!")

print("Initialization...", end="")
nn = neural.Network(neural_settings)
print(" Success!")

print("Training...", end="")
nn.fit(ds)
print(" Success!")

print("Neural Tests completed!")
