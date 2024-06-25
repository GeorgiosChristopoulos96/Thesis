import pandas as pd


def count_language_samples(file_path):
    # Load the TSV file
    data = pd.read_csv(file_path, delimiter=',', quotechar='"')

    # Extract the language information from the 'prefix' column
    language_counts = data['prefix'].value_counts()

    return language_counts


# Example usage:
file_path = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_MIXED(RU&EN)/combined_.tsv'
result = count_language_samples(file_path)
print(result)
