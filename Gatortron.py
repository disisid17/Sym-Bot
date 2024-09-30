from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer, AutoConfig

tokinizer= AutoTokenizer.from_pretrained('UFNLP/gatortron-medium')
config=AutoConfig.from_pretrained('UFNLP/gatortron-medium')
mymodel=AutoModel.from_pretrained('UFNLP/gatortron-medium')

encoded_input=tokinizer("Bone scan:  Negative for distant metastasis.", return_tensors="pt")
encoded_output = mymodel(**encoded_input)
print(encoded_output)
# then you can feed encoded_output to downstream task layers for different usecases e.g., NER, RE, MRC etc.