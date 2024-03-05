import pandas as pd
from transformers import pipeline, AutoTokenizer,AutoModelForSeq2SeqLM

text = "<S> Estádio Municipal Coaracy da Mata Fonseca <P> location <O> Arapiraca"
tokenizer = AutoTokenizer.from_pretrained("/Users/georgioschristopoulos/PycharmProjects/Thesis/rdf_to_text_model_final")
inputs = tokenizer(text, return_tensors="pt").input_ids


model = AutoModelForSeq2SeqLM.from_pretrained("/Users/georgioschristopoulos/PycharmProjects/Thesis/rdf_to_text_model_final")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

t = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))