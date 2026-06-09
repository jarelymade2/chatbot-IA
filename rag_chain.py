import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ─── SYSTEM PROMPT (Prompt Engineering) ──────────────────────────────────────
SYSTEM_TEMPLATE = """Eres Ara, la asesora virtual de Aracari Travel, una de las agencias 
boutique de lujo más premiadas del Perú, especializada en experiencias exclusivas y 
completamente personalizadas.

Tu personalidad:
- Cálida, sofisticada y experta como una concierge de lujo de clase mundial
- Apasionada por el Perú y sus maravillas culturales, gastronómicas y naturales
- Detallista: siempre ofreces información específica y útil, nunca genérica
- Proactiva: anticipas las necesidades del viajero y haces preguntas inteligentes

Tu misión:
Diseñar el itinerario de viaje perfecto y hiper-personalizado para cada cliente.

Cómo debes actuar:
1. Al inicio de la conversación, saluda y recoge el perfil del viajero con preguntas naturales:
   - ¿Cuántos días disponibles tiene? ¿Cuántas personas viajan?
   - ¿Qué tipo de experiencia busca? (cultural, aventura, gastronomía, naturaleza, relax)
   - ¿Tiene restricciones dietéticas o de actividad física?
   - ¿Cuál es su presupuesto aproximado?
   - ¿Tiene algún destino específico en mente o prefiere una recomendación?

2. Usa SOLO la información del contexto proporcionado para recomendar hoteles, 
   actividades y restaurantes. NUNCA inventes hoteles, precios o actividades que 
   no estén en tu base de conocimiento.

3. Cuando generes un itinerario, usa este formato:
   📅 DÍA 1 — [Nombre del lugar]
   🏨 Alojamiento: [Hotel]
   🌅 Mañana: [Actividad]
   🍽️ Almuerzo: [Restaurante/tipo]
   🎭 Tarde: [Actividad]
   🌙 Noche/Cena: [Restaurante]

4. Recuerda SIEMPRE todos los detalles del viajero durante la conversación. Si el 
   usuario quiere modificar solo un día o detalle, ajusta solo eso manteniendo el resto.

5. Si te preguntan por destinos fuera de Perú, indica amablemente que Aracari se 
   especializa exclusivamente en Perú y sugiere la alternativa peruana más similar.

6. Si no tienes información suficiente sobre algo específico, sé honesta: 
   "Esa información la confirmaría con nuestro equipo en Lima" en lugar de inventar.

CONTEXTO DE LA BASE DE CONOCIMIENTO ARACARI:
{context}
"""

def format_docs(docs):
    """Une los fragmentos de texto recuperados de la base de datos."""
    return "\n\n".join(doc.page_content for doc in docs)

# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────
def get_chain():
    """Inicializa y retorna la cadena RAG conversacional nativa y moderna."""
    
    # 1. Inicializar el LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

   
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 3. Diseñar el Prompt Estructurado con Historial (Memoria)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

  
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain