import json

# Load your JSON data from a file or any other source
with open('/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/aggregated_train.json', 'r') as f:
    data = json.load(f)

# Initialize the maximum length for English and French
max_length_en = 0
max_length_fr = 0
max_length_am = 0
max_length_ar = 0
max_length_ga = 0
max_length_gd = 0
max_length_it = 0
max_length_he = 0

total_length_en = 0
total_length_fr = 0
total_length_am = 0
total_length_ar = 0
total_length_ga = 0
total_length_gd = 0
total_length_it = 0
total_length_he = 0

count_en = 0
count_fr = 0
count_am = 0
count_ar = 0
count_ga = 0
count_gd = 0
count_it = 0
count_he = 0
# Iterate through the JSON objects
for translation_item in data["translation"]:
            # Check for "en" key
            if "en" in translation_item:
                max_length_en = max(max_length_en, len(translation_item["en"]))
                total_length_en += len(translation_item["en"])
                count_en += 1
            # Check for "fr" key
            if "fr" in translation_item:
                max_length_fr = max(max_length_fr, len(translation_item["fr"]))
                total_length_fr += len(translation_item["fr"])
                count_fr += 1
            if "am" in translation_item:
                max_length_am = max(max_length_am, len(translation_item["am"]))
                total_length_am += len(translation_item["am"])
                count_am += 1
            if "ar" in translation_item:
                max_length_ar = max(max_length_ar, len(translation_item["ar"]))
                total_length_ar += len(translation_item["ar"])
                count_ar += 1
            if "ga" in translation_item:
                max_length_ga = max(max_length_ga, len(translation_item["ga"]))
                total_length_ga += len(translation_item["ga"])
                count_ga += 1
            if "gd" in translation_item:
                max_length_gd = max(max_length_gd, len(translation_item["gd"]))
                total_length_gd += len(translation_item["gd"])
            if "it" in translation_item:
                max_length_it = max(max_length_it, len(translation_item["it"]))
                total_length_it += len(translation_item["it"])
                count_it += 1
            if "he" in translation_item:
                max_length_he = max(max_length_he, len(translation_item["he"]))
                total_length_he += len(translation_item["he"])
                count_he += 1


average_length_en = total_length_en / count_en if count_en > 0 else 0
average_length_fr = total_length_fr / count_fr if count_fr > 0 else 0
average_length_am = total_length_am / count_am if count_am > 0 else 0
average_length_ar = total_length_ar / count_ar if count_ar > 0 else 0
average_length_ga = total_length_ga / count_ga if count_ga > 0 else 0
average_length_gd = total_length_gd / count_gd if count_gd > 0 else 0
average_length_it = total_length_it / count_it if count_it > 0 else 0
average_length_he = total_length_he / count_he if count_he > 0 else 0
# Calculate averages for other languages if needed

print(f"The average length of English strings is: {average_length_en:.2f}")
print(f"The average length of French strings is: {average_length_fr:.2f}")
print(f"The average length of Amharic strings is: {average_length_am:.2f}")
print(f"The average length of Arabic strings is: {average_length_ar:.2f}")
print(f"The average length of Irish strings is: {average_length_ga:.2f}")
print(f"The average length of Scottish strings is: {average_length_gd:.2f}")
print(f"The average length of Italian strings is: {average_length_it:.2f}")
print(f"The average length of Hebrew strings is: {average_length_he:.2f}")

print(f"The max length of English strings is: {max_length_en:.2f}")
print(f"The max length of French strings is: {max_length_fr:.2f}")
print(f"The max length of Amharic strings is: {max_length_am:.2f}")
print(f"The max length of Arabic strings is: {max_length_ar:.2f}")
print(f"The max length of Irish strings is: {max_length_ga:.2f}")
print(f"The max length of Scottish strings is: {max_length_gd:.2f}")
print(f"The max length of Italian strings is: {max_length_it:.2f}")
print(f"The max length of Hebrew strings is: {max_length_he:.2f}")

