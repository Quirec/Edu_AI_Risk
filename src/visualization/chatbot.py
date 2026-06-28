import streamlit as st
from groq import Groq


MODEL_NAME = "llama-3.1-8b-instant"


def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def get_bot_response(messages):
    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4,
        max_tokens=500
    )

    return response.choices[0].message.content


def mostrar_chatbot():
    st.subheader("Chatbot Edu AI Risk")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente educativo llamado Edu AI Risk. "
                    "Ayudas a interpretar información sobre alumnos en riesgo de reprobar. "
                    "Explicas métricas de dashboards, resultados de modelos de clasificación "
                    "y sugieres acciones de apoyo académico de forma clara y breve. "
                    "No inventes datos específicos si el usuario no los proporciona."
                )
            }
        ]

    for msg in st.session_state.chat_messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.write(prompt)

        try:
            respuesta = get_bot_response(st.session_state.chat_messages)
        except Exception as e:
            respuesta = (
                "No pude conectarme con Groq. "
                "Revisa que tu API key esté bien guardada en "
                "`.streamlit/secrets.toml` y que no hayas superado los límites de uso."
            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": respuesta
            }
        )

        with st.chat_message("assistant"):
            st.write(respuesta)