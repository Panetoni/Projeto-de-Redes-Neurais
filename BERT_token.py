from transformers import AutoTokenizer, AutoModelForPreTraining

# Baixa e carrega o modelo BERT pré-treinado
modelo_nome = "neuralmind/bert-base-portuguese-cased"
model = AutoModelForPreTraining.from_pretrained(modelo_nome)
tokenizer = AutoTokenizer.from_pretrained(modelo_nome, do_lower_case=False)

# Texto de entrada
texto = "O BERT é um modelo muito poderoso para NLP."

# Tokenização do texto
tokens = tokenizer(texto, return_tensors="pt")  # Retorna tensores PyTorch

# Executa o modelo no texto tokenizado
output = model(**tokens)

# Exibe os tokens processados
print("\n\n\n")
print("Resultado dos tokens: ")
print(tokens)

