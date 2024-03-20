from transformers import AutoTokenizer
text = "ለእናንተና ለእንስሶቻችሁ መጣቀሚያ ይኾን ዘንድ ( ይህን አደረገ )"
tokenizer = AutoTokenizer.from_pretrained("/Users/georgioschristopoulos/PycharmProjects/Thesis/my_awesome_mt5_model")
inputs = tokenizer(text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("/Users/georgioschristopoulos/PycharmProjects/Thesis/my_awesome_mt5_model")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
l= tokenizer.decode(outputs[0], skip_special_tokens=True)
print(l)