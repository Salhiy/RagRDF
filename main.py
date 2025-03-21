from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForQuestionAnswering
from sparql.query_runner import load_rdf_graph, run_sparql
from bert_rag.bert_with_rag import answer_with_context
import torch

# Chargement du modèle Mistral fine-tuné avec LoRA
mistral_path = "./models/mistral_sparql_lora"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_path, trust_remote_code=True)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_path, device_map="auto", trust_remote_code=True)

# Chargement du modèle BERT pour l'étape RAG
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Chargement du graphe RDF local
rdf_graph = load_rdf_graph("data/rdf_graph.ttl")

# Génère une requête SPARQL à partir d'une question
def generate_sparql(question):
    prompt = f"Tu es un agent intelligent. Génère une requête SPARQL pour répondre à la question suivante : {question}\nRequête SPARQL : "
    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral_model.device)
    outputs = mistral_model.generate(**inputs, max_new_tokens=256)
    sparql_query = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Requête SPARQL :" in sparql_query:
        sparql_query = sparql_query.split("Requête SPARQL :")[-1].strip()
    return sparql_query

# Pipeline RAG complet
def rag_pipeline(question):
    print(f"\n❓ Question : {question}")

    sparql_query = generate_sparql(question)
    print(f"\n📡 Requête SPARQL générée :\n{sparql_query}")

    try:
        results = run_sparql(sparql_query, rdf_graph)
        if not results:
            return "❌ Aucune donnée trouvée dans le graphe RDF."
        rdf_context = " ".join(results)
        print(f"\n📘 Résultats RDF : {rdf_context}")
    except Exception as e:
        return f"⚠️ Erreur SPARQL : {e}"

    answer = answer_with_context(question, rdf_context)
    return f"\n✅ Réponse finale : {answer}"

# Exemple
if __name__ == "__main__":
    question = "Quels moteurs équipe le Boeing 737 ?"
    final_answer = rag_pipeline(question)
    print(final_answer)
