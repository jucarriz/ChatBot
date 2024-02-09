import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.text_splitter import MarkdownHeaderTextSplitter
import pandas as pd
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import numpy as np
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
import base64
import streamlit as st
from PIL import Image

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

chat = AzureChatOpenAI(temperature=0, deployment_name="chat")

#Interfaz

# pestaña
st.set_page_config(page_title="Asistente Galicia", page_icon="🤖", layout="centered", initial_sidebar_state="auto", menu_items=None)

# fondo
def set_background(path: str) -> None:
    """This functions set the app backgorund."""
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\Fondo_galicia")

# logo
def set_logo(path: str) -> None:
    """This function"""
    image = Image.open(path)
    st.image(
        image, 
        use_column_width=False, 
        width=int(image.size[1] * 1.235),
        output_format='PNG'
    )
set_logo(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\logo_galicia.png")

# bienvenida
st.markdown('<h1 style="color: white; text-align: center;">¡Bienvenido al portal de consultas del Banco Galicia!</h1>', unsafe_allow_html=True)
html_style= '''<style>div.st-emotion-cache-7sak6c{padding-bottom: 1rem;}</style>'''
st.markdown(html_style, unsafe_allow_html=True)


# lo que se hizo fue: tomar un pdf,gcp, lo tomo lo dividio en chunks, le calculo los embeddings y lo alamceno en archivo pickle, se lo calculo los embeddings con sentece_stransformer.
# la idea era que el usuario haga una pregunta, en base a esa pregunta se le calculen los embeddings, hiciera busqueda semantica, con todos los chunks del archivo.pickle.
# la idea era q devuelva los 5 mas similares y esos 5 se los paso al modelo
data_embeddings = pd.read_pickle(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\CG_Mascotas.pickle") # ver si poner csv o hacer que ese csv sea un libro binario de excel

# PDF
document = PyPDFLoader(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\CG_Mascotas.pdf")
data = document.load()
text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = 700,
         length_function = len,
         chunk_overlap = 250
 )
documents = text_splitter.split_documents(data)

# TXT
# document = TextLoader(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\audioprueba.txt",encoding='utf-8').load()
# text_splitter = CharacterTextSplitter(chunk_size = 250,chunk_overlap = 50)
# documents = text_splitter.split_documents(document)

#MARKDOWN
# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3")
# ]   

# # Crear una instancia del MarkdownHeaderTextSplitter
# splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# # Abrir y leer el archivo de texto
# # with open(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\Seguro Vida Bienestar2.txt", 'r',encoding = 'utf-8') as file:
# #     texto = file.read()]
# texto = TextLoader(r"C:\Users\jucarriz\OneDrive - NTT DATA EMEAL\Attachments\Equipo Genai\Formación enero\Seguro Vida Bienestar2.txt",encoding='utf-8').load()
# # Dividir el texto en secciones utilizando encabezados Markdown
# secciones = splitter.split_text(texto[0].page_content)


# Función que retorna una lista de las páginas sin repetir.
def pages_number(pags:list):
    list_pages = []
    for pag in range(len(pags)):
        list_pages.append(pags[pag]["page"])
    list_pages = list(set(list_pages))
    return list_pages

# Función que calcula el producto punto entre dos vectores
def distancia_vector(x, y):
    return np.dot(np.array(x), np.array(y))

# Función que retorna un dataframe con la pregunta del usuario junto a su vector.
def question_user(q:str,embeddings=embedding_function):
  data_question = pd.DataFrame()
  emb = []
  q_list = []
  emb.append(embeddings.embed_query(q))
  q_list.append(q)
  data_question["pregunta"] = q_list
  data_question["embedding_pregunta"] = emb

  return data_question

# Función que retorna el mismo dataframe ingresado, pero con una columna más,
# que es la distancia entre el vector del usuario con todos los vectores almacenados del documento.
def data_metadata(data:object, p:str):
    data_p = question_user(p)
    data["distancia"] = data["embeddings"].apply(lambda x:distancia_vector(data_p.iloc[0,1],x))
    return data, p


# Función que retorna una lista ordenada de forma descendente y que está filtrada
# # por el parámetro ingresado por el usuario, además retorna la pregunta del usuario
def metadata_final(data:object, p:str, param:float):
    data_sorted = data.sort_values(by = "distancia",ascending=False)
    # ver de hacer un print, ahi aparece el mayor y se ve la variación
    data_sorted = data_sorted[data_sorted["distancia"] >= param]
    content = data_sorted["metadata"].tolist()
    return content, p

# Función que invoca a las funciones anteriores.
def function_main_content(p:str, data:object):
    data, p = data_metadata(data, p)
    content, p = metadata_final(data, p, 0.45) # jugar y ver de ampliarlo, a la hora de ver la respuesta.
    return content, p

# Función que retorna los documentos que cumplen la condición del parámetro, 
# estos son los documentos  
def documents_prompt(documents:list, pages:list):
    docs = []
    for doc in range(len(documents)):
        val_aux = documents[doc].metadata["page"]
        if val_aux in pages:
            docs.append(documents[doc])
        else:
            continue
    return docs

# Función que invoca a las funciones anteriores y que retorma los documentos finales
# que va a recibir el LLM, además de retornar las páginas correspondientes a los documentos.
def documents_main(p:str, data:object, documents:list):
    pags, p = function_main_content(p,data)
    pags = pages_number(pags)
    docs = documents_prompt(documents, pags)
    pags = sorted(pags)
    return docs, pags   


#@st.cache_data
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
def conversation_complete(query:str, chat=chat, documents=documents,
                           data_embeddings=data_embeddings):
    
    try:
        # Definición del prompt
        #template = """
        #Contexto de la conversación {chat_history}
        #Eres un experto en contratos de prestación de servicios y 
        #tienes la siguiente información para interpretar: {context}
        #y debes responder la pregunta:{human_input}
        #Responde solo en español y utilizando solo la información que recibes.
        #Contesta de forma muy detallada."""
        template = """

        contexto de la conversación: {chat_history} \
        
        eres Gali, un asistente amable y respetuoso del Banco Galicia y

        tienes la siguiente información para interpretar: {context}

        y debes responder la pregunta: {human_input}\

        responde solo en español y utilizando la información que recibes.\

        contesta de clara y detalladamente a como está en el documento original.\
        
        """

        prompt = PromptTemplate(
                    input_variables=["chat_history", "human_input", "context"], template=template
            )
        # memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        # Se crea la cadena
        chain = load_qa_chain(
        chat, chain_type="stuff", memory=memory, prompt=prompt)
        docs_, pags = documents_main(query,data_embeddings, documents) # Se obtienen los documentos filtrados con sus páginas correspondientes.
        # print("DOCUMENTOS:",docs_)
        
        # Se obtiene la respuesta del asistente
        respuesta = chain({"input_documents": docs_[:10], "human_input": query}, return_only_outputs=True)
        # Se captura la salida final
        respuesta_final = respuesta['output_text']
        
        # Se válida la salida del asistente.
        if query == '\n':
            respuesta_final = "Lo siento, puedes reformular la pregunta."
            pags = []
            return respuesta_final
            
        elif docs_ == []:
            respuesta_final = "Su pregunta no se relaciona con la información del documento"
            pags = []
            return respuesta_final
        
        elif query == 'no, gracias' or query == 'no' or query == 'no gracias' or query == 'no por el momento':
            respuesta_final = "Espero haber resuelto tus dudas e inquietudes, que tengas buen dia! Gracias por haberte comunicado con el Banco Galicia."
            # ver quizas de validar la respuesta
            return respuesta_final # ver como hacer para que el asistente deje de andar despues de esto.
        else:
            return respuesta_final
    except Exception as e:
        pags = []
        respuesta_final = "Lo siento, pero tu pregunta me provocó confusión, ¿Puedes reiniciarme?"
        return respuesta_final

# Metricas
    
# Precisión
preguntas_respuestas_esperadas = {
    "Hablame de las condiciones generales del seguro vida bienestar":"El seguro Vida Bienestar cubre al asegurado sin restricciones en cuanto a residencia y viajes dentro o fuera del país. Sin embargo, existen ciertas causas por las cuales la compañía no pagará la indemnización en caso de fallecimiento del asegurado. Estas causas incluyen el suicidio voluntario del asegurado, actos ilícitos del beneficiario que provoquen deliberadamente la muerte del asegurado, duelos o riñas que no sean en legítima defensa, actos o hechos de guerra civil o internacional, guerrilla, rebelión, insurrección, entre otros. Además, no se cubren situaciones como el abuso de alcohol o drogas, intervenciones médicas o quirúrgicas ilícitas, participación en actividades deportivas peligrosas, entre otros.",
    "De qué trata el item C del ANEXO I - RIESGOS NO CUBIERTOS?":"El ANEXO I - RIESGOS NO CUBIERTOS trata sobre las causas por las cuales la Compañía no pagará la indemnización en caso de fallecimiento del Asegurado.",
}
total_preguntas = len(preguntas_respuestas_esperadas)
 

#errores = 0
#total_preguntas = len(preguntas_respuestas_esperadas)

#for pregunta, respuesta_esperada in preguntas_respuestas_esperadas.items():
#     respuesta_asistente = respuestas_asistente.get(pregunta)
#     if respuesta_asistente != respuesta_esperada:
#         errores += 1


# 
# tasa_error = errores / total_preguntas
# print(f"Tasa de Error de Conversación (CER): {tasa_error:.2f}") va a tener que ser un st.write

# Estilo personalizado
# font-size = regula letra.
# padding el alto del recuadro
# color color de la letra
# font-weight 


# incializo la chat_history
primera_vez = True
if primera_vez:
    with st.chat_message(name = "assistant",avatar = "👨‍💼"): #🤖👨‍💼
        bienvenida = st.write("Hola, soy Gali, el asistente virtual del Banco Galicia, ¿en qué puedo ayudarle?")
    primera_vez = False

if "messages" not in st.session_state: # el bot responde lo q le mandas. Esto ayuda a q recuerde la informacion entre las interacciones. Si uno existen una serie de mensajes la lista esta vacia
    st.session_state.messages = [] # por cada conversacion, que forma parte del elemento de cada lista, se forma de la siguiente manera {role:"user","content:"our prompt"}, {role:"assistant","content:"the response"}

# muestro los mensajes del chat de la historia en la interfaz.
for message in st.session_state.messages: # si yo corro esto y lo anterior, no aparece nada porque la lista esta vacia
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pregunta := st.chat_input("Ingresa una pregunta: "):
    # muestro el mensaje del usuario en el contenedor del mensaje lo guardo en la lista
    with st.chat_message("user",avatar = "🙍‍♂️"): #👥
        st.markdown(pregunta)
    st.session_state.messages.append({"role":"user","content":pregunta})
    respuesta = conversation_complete(pregunta)
    # muestro la respuesta del asistente en el contenedor del mensaje y lo guardo en la lista
    with st.chat_message("assitant",avatar = "👨‍💼"):
        st.markdown(respuesta)
    st.session_state.messages.append({"role":"assistant","content":respuesta})
    primera_vez = False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # respuestas_asistente = {
    # "Hablame de las condiciones generales del seguro vida bienestar":conversation_complete("Hablame de las condiciones generales del seguro vida bienestar"), # no se si ponerle respuesta directo al valor
    # "De qué trata el item C del ANEXO I - RIESGOS NO CUBIERTOS?":conversation_complete("De qué trata el item C del ANEXO I - RIESGOS NO CUBIERTOS?") # no se si ponerle respuesta directo al valor
    # }
    # errores = 0
    # for buscador in respuestas_asistente.keys():
    #     if (buscador == pregunta):
    #         for pregunta, respuesta_esperada in preguntas_respuestas_esperadas.items():
    #             respuesta_asistente = respuestas_asistente.get(pregunta)
    #             if respuesta_asistente != respuesta_esperada:
    #                 errores += 1
    # with st.chat_message("assitant",avatar = "👨‍💼"):
    #     tasa_error = errores / total_preguntas
    #     st.write("La precisión fue de: ",tasa_error)
    # st.session_state.messages.append({"role":"assistant","content":respuesta})



#QUEDARIA VER DE PASAR LOS PDFS A TEXTOS

#pregunta = st.text_input("Ingresa una pregunta: ")
#respuesta = conversation_complete(pregunta)
#st.write(respuesta)

#conversacion.append(("usuario",pregunta))
#conversacion.append(("chatbot",respuesta))

#for remitente, mensaje in conversacion:
 #   if remitente == 'usuario':
  #      st.text_input('Tú:', mensaje)
   # else:
    #    st.text_area('Chatbot:', mensaje, height=100)
# JUNTARSE CON EL GRUPO PARA VER ALCANCES Y OBJETIVOS DEL ASISTENTE, CUALES VAN A SER LAS LIMTAICIONES, RECURSOS COMO STREAMLIT.
# HAY Q VISUALIZAR EL FLUJO, EL ALCANCE, LAS LIMITACIONES.
# COMO RESPONDE TU ASISTENTE, AMABLE O RUDO
# COBERTURA, Q EVALUAMOS, EL MODELO?, SI EVALUAMOS CON EVAL_QUA
# ES IMPORTANTE COMO TENEMOS LOS DATOS, COMO TRATARLOS. 
# MIENTRAS MAS CONICSA LA INFO Q SE LE PASE AL PROMPT, MEJOR. 

