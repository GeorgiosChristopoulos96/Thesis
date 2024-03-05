text = "translate English to French: I love you"


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model/checkpoint-12500")
inputs = tokenizer(text, return_tensors="pt").input_ids


from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model/checkpoint-12500")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

tokenizer.decode(outputs[0], skip_special_tokens=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))