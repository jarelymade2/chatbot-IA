import streamlit as st
from rag_chain import get_chain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Ara - Aracari Travel", page_icon="🦙", layout="centered")

st.title("🦙 Consulta con Ara — Asesora de Viajes")
st.subheader("Experiencias boutique de lujo en el Perú")

# Inicialización de la memoria en la sesión de Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    with st.spinner("Inicializando sistemas de IA y base de datos..."):
        try:
            st.session_state.rag_chain = get_chain()
        except Exception as e:
            st.error(f"Error al inicializar la cadena de IA: {e}")

# Renderizar el historial de conversación en la interfaz
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Caja de texto para la entrada de usuario
if user_query := st.chat_input("¿En qué puedo ayudarte a planificar tu viaje hoy?"):
    
    # Mostrar el mensaje actual del usuario en pantalla
    with st.chat_message("user"):
        st.write(user_query)
        
    if "rag_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Ara está procesando tu solicitud..."):
                try:
                    # Ejecutar la cadena pasando las variables de forma directa
                    respuesta_texto = st.session_state.rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    st.write(respuesta_texto)
                    
                    # Guardar el intercambio de mensajes en el historial
                    st.session_state.chat_history.extend([
                        HumanMessage(content=user_query),
                        AIMessage(content=respuesta_texto)
                    ])
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar tu consulta: {e}")