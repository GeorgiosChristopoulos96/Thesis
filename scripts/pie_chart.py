import json
from collections import Counter
import matplotlib.pyplot as plt

PATH = '/Users/georgioschristopoulos/PycharmProjects/Thesis'
# Load your JSON data for train, test, and validation
train_data = json.load(open(f'{PATH}/Datasets/OPUS-100/test_augmented/aggregated_train.json', 'r'))
test_data = json.load(open(f'{PATH}/Datasets/OPUS-100/test_augmented/aggregated_test.json', 'r'))
validation_data = json.load(open(f'{PATH}/Datasets/OPUS-100/test_augmented/aggregated_validation.json', 'r'))

# Assuming each dataset has a key named 'translation'
translation_key = 'translation'

# Assuming each dataset has a key named 'translation'
translation_key = 'translation'

# Function to count PARENT_metric occurrences in a dataset
def count_languages(dataset):
    language_counter = Counter()

    for item in dataset.get(translation_key, []):
        # Exclude the "id" key from languages
        languages = [key for key in item.keys() if key != "id"]
        language_pair = "-".join(sorted(languages))  # Create PARENT_metric pair (e.g., "en-am")
        language_counter.update([language_pair])

    return language_counter

# Count languages for each dataset
train_languages = count_languages(train_data)
test_languages = count_languages(test_data)
validation_languages = count_languages(validation_data)

# Function to create a pie chart
def create_pie_chart(language_counter, title):
    labels = list(language_counter.keys())
    sizes = list(language_counter.values())
    total_items = sum(sizes)
    percentages = [(size / total_items) * 100 for size in sizes]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.title(f'{title} - Total Items: {total_items}')

    plt.show()

# Create pie charts for each dataset
create_pie_chart(train_languages, 'Train Dataset')
create_pie_chart(test_languages, 'Test Dataset')
create_pie_chart(validation_languages, 'Validation Dataset')