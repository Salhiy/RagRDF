import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

paraphrase_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.8,
)

def generate_paraphrase(question):
    prompt = f"""
Tu es un agent de paraphrase intelligent. Réécris la phrase suivante en changeant certains mots et la structure, mais en gardant exactement le même sens.

Phrase originale : "{question}"

Réécris la phrase :"""
    
    result = paraphrase_generator(prompt)[0]["generated_text"]

    if "Réécris la phrase :" in result:
        paraphrased = result.split("Réécris la phrase :")[-1].strip()
        if "\n" in paraphrased:
            paraphrased = paraphrased.split("\n")[0]
        return paraphrased
    return question  # fallback si besoin

def drop_random_word(question):
    words = question.split()
    if len(words) > 4:
        idx = random.randint(0, len(words) - 1)
        del words[idx]
    return " ".join(words)

def swap_words(question):
    words = question.split()
    if len(words) > 3:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def augment_question_full(original_q):
    variations = set()

    variations.add(original_q)

    variations.add(drop_random_word(original_q))

    variations.add(swap_words(original_q))

    try:
        paraphrased = generate_paraphrase(original_q)
        variations.add(paraphrased)
    except Exception as e:
        print(f"Erreur paraphrase : {e}")

    return list(variations)

def generate_augmented_dataset(input_csv="data/qa_dataset.csv", output_csv="data/augmented_dataset.csv"):
    df = pd.read_csv(input_csv)
    all_rows = []

    for _, row in df.iterrows():
        original_q = row["question"]
        sparql = row["sparql"]

        # Génère toutes les variantes
        variations = augment_question_full(original_q)

        for q in variations:
            all_rows.append({"question": q, "sparql": sparql})

    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    print(f"Dataset augmenté enregistré dans {output_csv}")