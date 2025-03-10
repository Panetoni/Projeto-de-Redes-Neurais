import torch
from transformers import AutoTokenizer, AutoModel

# Carregar modelo e tokenizer
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModel.from_pretrained(model_name)

# Texto de entrada
texto = "Aprender BERT em português é muito interessante."

# Tokenizar e converter para tensor
tokens = tokenizer(texto, return_tensors="pt")

# Passar pelo modelo
with torch.no_grad():
    outputs = model(**tokens)

# Extraindo os embeddings da última camada
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # (1, número de tokens, dimensão do embedding)
print(embeddings)

