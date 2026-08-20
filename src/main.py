import streamlit as st
import os
import json
import joblib
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# Кешування моделей
@st.cache_resource
def load_artifacts():
    print("🚀 Завантаження моделей та індексів...")
    
    config_path = os.path.join("models", "config.json")
    model_path = os.path.join("models", "classifier_model.joblib")
    faiss_path = os.path.join("models", "faiss_index.faiss")
    texts_path = os.path.join("data", "processed", "rag_texts.pkl")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        threshold = config['threshold']
        model_name = config['model_name']
    
    encoder = SentenceTransformer(model_name, device='cuda') # або 'cpu'
    classifier = joblib.load(model_path)
    faiss_index = faiss.read_index(faiss_path)
    
    with open(texts_path, 'rb') as f:
        rag_texts = pickle.load(f)
    
    print("Всі артефакти завантажено!")
    return encoder, classifier, threshold, faiss_index, rag_texts

# Завантажуємо один раз 
encoder, classifier, threshold, faiss_index, rag_texts = load_artifacts()

# Інтерфейс користувача (UI) 
st.set_page_config(page_title="ToxicGuard-RAG", layout="centered")
st.title("ToxicGuard-RAG")
st.write("Введіть коментар українською, російською або суржиком, щоб перевірити його на токсичність.")

# Поле для вводу
text_input = st.text_area("Введіть текст коментаря:", height=100)

# Кнопка запуску
if st.button("🔍 Проаналізувати"):
    if not text_input.strip():
        st.warning("Будь ласка, введіть текст перед аналізом.")
    else:
        with st.spinner("Аналізую коментар та шукаю схожі випадки..."):
            # 1. Ембеддинг
            new_embedding = encoder.encode([text_input]).astype('float32')
            
            # Класифікація
            proba = classifier.predict_proba(new_embedding)[0][1]
            is_toxic = int(proba >= threshold)
            
            result = {
                "text": text_input,
                "toxic": is_toxic,
                "confidence": float(proba),
                "explanation": "Коментар безпечний."
            }

            # Якщо токсичний — RAG + Ollama
            if is_toxic == 1:
                # Пошук 3 схожих
                distances, indices = faiss_index.search(new_embedding, k=3)
                similar_texts = [rag_texts[i] for i in indices[0]]

                result["similar_texts"] = similar_texts
                
                prompt = (
                    f"Ти - модератор контенту. Користувач написав: '{text_input}'. "
                    f"Ось 3 схожі заблоковані коментарі з історії: {similar_texts}. "
                    "Поясни коротко, чому новий коментар схожий на ці приклади і чому його варто заблокувати. "
                    "Відповідай українською."
                )
                
                try:
                    # Збільшуємо таймаут, як ми обговорювали раніше
                    ollama_response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
                        timeout=120
                    )
                    if ollama_response.status_code == 200:
                        result["explanation"] = ollama_response.json()["response"]
                    else:
                        result["explanation"] = "Помилка зв'язку з Ollama (статус {ollama_response.status_code})."
                except requests.exceptions.ConnectionError:
                    result["explanation"] = "Помилка: Ollama не запущена. Запустіть 'ollama serve' або 'ollama run qwen2.5:7b'."
                except Exception as e:
                    result["explanation"] = f"Помилка генерації: {str(e)}"

            # --- 4. Відображення результатів ---
            st.divider()
            
            if result["toxic"] == 1:
                st.error(f"Токсичний коментар! (Впевненість: {result['confidence']:.2%})")
            else:
                st.success(f"Безпечний коментар. (Впевненість: {result['confidence']:.2%})")
            
            st.write(f"Пояснення: {result['explanation']}")
            if result.get("similar_texts"):
                st.markdown("Схожі заблоковані коментарі з бази:")
                for i, comment in enumerate(result["similar_texts"]):
                    st.info(f"Приклад {i+1}: {comment}")
            
            with st.expander("Деталі запиту (JSON)"):
                st.json(result)