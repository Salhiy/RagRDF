from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

def answer_with_context(question, rdf_context):
    
    prompt_context = (
        "Tu es un agent intelligent RAG qui répond à des questions sur l'aviation. "
        "Tu utilises des données structurées extraites d’un graphe RDF, via des requêtes SPARQL.\n\n"
        f"Contexte RDF : {rdf_context}\n\n"
        f"Question : {question}"
    )

    inputs = tokenizer(question, prompt_context, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)

    if start <= end:
        answer = tokenizer.decode(inputs["input_ids"][0][start:end + 1])
    else:
        answer = "[Aucune réponse trouvée]"

    return answer
