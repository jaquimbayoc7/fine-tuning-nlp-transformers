

Maestría Virtual en Ingeniería de Sistemas y Computación

Gran idea 2: Taller de clasificación de texto y NER

Grandes lenguajes de modelamiento para  la tarea de clasificación


Los LLMs (Large Language Models) se utilizan actualmente para tareas de
procesamiento inteligente de texto. La capacidad contextual de estos modelos
supera ampliamente a la de los enfoques de etiquetado secuencial y los algoritmos
tradicionales de machine learning. Además, una de las tareas fundamentales del
Procesamiento del Lenguaje Natural (PLN) es la clasificación de texto,
especialmente el análisis de sentimientos. El objetivo de este taller es implementar
un clasificador binario y un clasificador de tres clases  (análisis de sentimientos),
además de implementar un  modelo de Reconocimiento de Entidades Nombradas
(NER) utilizando LLMs. Cada tarea se divide en cuatro fases:
- Preprocesamiento de los conjuntos de datos,
## 2. Tokenización,
- Entrenamiento o finetuning del modelo, y
- Evaluación del desempeño y publicación del modelo en Hugging Face.


## BERT

El modelo BERT es una arquitectura Transformer-encoder preentrenada sobre un
corpus extenso de datos en inglés y multilingüe, mediante aprendizaje
auto-supervisado. Esto significa que se entrenó únicamente con texto sin etiquetar
(sin intervención humana para generar etiquetas), lo que permite aprovechar
grandes volúmenes de datos públicos. Su preentrenamiento se basa en dos
objetivos principales:

● Modelado de lenguaje enmascarado (MLM):
Toma una oración, enmascara
aleatoriamente el 15 % de las palabras en la entrada, la procesa completa a través
del modelo y predice las palabras enmascaradas. A diferencia de las redes
neuronales recurrentes tradicionales (RNN), que procesan las palabras
secuencialmente, o de los modelos autorregresivos como GPT, que enmascaran los
tokens futuros, MLM permite que el modelo aprenda una representación
bidireccional de la oración.


● Predicción de la siguiente oración (NSP): El modelo concatena dos oraciones
enmascaradas como entradas durante el entrenamiento previo. A veces corresponden a
frases que estaban una al lado de la otra en el texto original, a veces no. Luego, el modelo
tiene que predecir si las dos oraciones se suceden o no. De esta manera, el modelo
aprende una representación interna del idioma inglés que luego puede usarse para extraer
## 1

características útiles para tareas posteriores: si tiene un conjunto de datos de oraciones
etiquetadas, por ejemplo, puede entrenar un clasificador estándar usando las características
producidas por BERT.
En la siguiente figura se presenta el esquema de finetuning o
ajuste, donde a partir de un gran corpus de preentrenamiento se adapta un conjunto
de datos anotado con el fin de transferir el modelo preentrenado y resolver una tarea
específica de PLN.





Punto 1. Clasificación de texto usando el dataset en español de TASS  y los  LLMs de
XLnet, XLM-RoBERTa

En este ejercicio se deben afinar y entrenar dos conjuntos de datos:
## TASS,

(anotado con tres clases: NEG, NEU, POS) y
## Sarcasmo
(anotado con dos clases:
sarcasmo y no_sarcasmo).
Para ello se utilizarán  los  modelos de clasificación  BETO,
XLNet y XML-Roberta-large.

Esta tarea consta de dos subtareas. En la primera, se debe afinar el dataset
## TASS

con los modelo
BETO,  XLNet y XML-Roberta-large siguiendo estos pasos:

1.Cargar  el dataset   TASS de manera local y  mostrar la distribución de etiquetas para los
tres conjuntos
(entrenamiento, validación y prueba).

-  Implementar el ajuste o fine tuning para los modelos  Xlnet y XML-Roberta-large sobre el
dataset  TASS considerando todas las fases; preprocesamiento,  alineación  del dataset,
tokenización, padding (mapeo)  y el entrenamiento. Como insumo, se comparte el finetuning
del modelo BETO sobre  TASS el cual alcanza un f1-score del 64% . La idea es superar
éste rendimiento con   Xlnet y XML-Roberta-large.
## 2




De igual manera, se debe afinar y  entrenar el dataset  Sarcasmo  con  los modelos de
BETO y XML-Roberta-large implementando los mismos pasos del anterior
(preprocesamiento,  alineación, tokenización, padding  y el entrenamiento).  En este sentido,
es importante probar varias combinaciones de batch size según la capacidad  de la GPU (8,
16 y 32 GB),  utilizando GPU de  Colab o Kaggle  para ambos afinamientos.
-  Implementar  las métricas de  desempeño (f1-score) para batch size de 8,16, 32
ejecutando entre 6 y 10 épocas con  Early Stopping para cada uno de los  dos afinamientos
de TASS y Sarcasmo  con los respectivos modelos.

- Los modelos se deben publicarse en Hugging face y utilizarse para clasificar las
siguientes sentencias:




Para probar los modelos entrenados con el dataset Sarcasmo, usar los siguientes ejemplos.

Texto: ¡Qué divertido, otro lunes de trabajo! 
Predicción: SARCASMO
## Confianza: 100.00%
Prob. sarcasmo: 100.00%

Texto: Me encanta cuando llueve el día que tengo planes al aire libre
Predicción: NO_SARCASMO
## 3

## Confianza: 99.86%
Prob. sarcasmo: 0.14%

Texto: Obvio que aprobé el examen, estudié exactamente 5 minutos
Predicción: SARCASMO
## Confianza: 100.00%
Prob. sarcasmo: 100.00%

Texto: Qué sorpresa que el internet se cayó justo cuando iba a entregar la
tarea
Predicción: SARCASMO
## Confianza: 100.00%
Prob. sarcasmo: 100.00%

Texto: Sí, porque claramente yo soy el experto en cocina gourmet ޽
Predicción: SARCASMO
## Confianza: 100.00%
Prob. sarcasmo: 100.00%

- Realizar un análisis completo respondiendo:
¿Por qué los rendimientos sobre  TASS  son menores  que los obtenidos por el dataset
## Sarcasmo
## ?.
¿Qué papel juegan el número de clases en ambos  datasets?
¿Analizar y comprar  el desempeño de los LLMs; Xlnet y XML-Roberta-large además de
BETO frente a  XML-Roberta-large   sobre el dataset Sarcasmo
## ?


Punto 2. Reconocimiento de entidades nombradas   (NER)

Esta tarea consiste en el reconocimiento de entidades nombradas (NER)  mediante
Machine Learning (usando LSTMs) sobre el dataset  Conll2002   y mediante  Transfer
Learning usando  LLMs bajo el afinamiento (finetuning) de modelos preentrenados tales
como   BETO y XML-Roberta sobre el dataset prostata. Como insumo para el ajuste y
entrenamiento de los LLMs, se proporciona en la carpeta   ayuda  un modelo base de
Bilstm-base con un  64.3% de f1-score y el modelo bert-base-cased con un 75% sobre
## Conll2002.

2.1 (NER con BiLSTMS)

En ésta primera parte, la idea es entrenar dos arquitecturas

1) BiLSTM + CRF +fastext
2) BILSTM + CRF + CNN +fastatex

Descarga de Fasttext:

## 4


El entrenamiento se realizará sobre el dataset Conll2002, partiendo del modelo base
Bilstm-base + Conll2002 . El objetivo es mejorar progresivamente el rendimiento de las tres
arquitecturas de LSTM  (contando el modelo base Bilstm-base).  Por ejemplo, para el
modelo Bilstm-base el f1-score es de 64.3%; a continuación se muestran las métricas de
desempeño sobre los datos de prueba o testeo de Conll2002.

En tu notebook BiLSTM-CRF+CNN, la  idea es que el  CNN (Convolutional Neural
Network) actúe como un extractor de rasgos locales para enriquecer las
representaciones de cada palabra en un vector de tamaño fijo, capturando así su
estructura morfológica. Este vector debe concatenarse con el embedding de la
palabra completa y alimentar el BiLSTM. Como se observa en la figura del código,
el vector de caracteres se suma al vector de embeddings de palabras. Para cada
palabra se combina el embeddings de Fasttext con el embedding de caracteres.

# Embeddings de palabra
self.word_emb = nn.Embedding.from_pretrained(
torch.tensor(emb_matrix), freeze=False, padding_idx=0
## )

# CharCNN
self.char_cnn = CharCNN(char_vocab_size, char_emb_dim,
num_filters=
## 30,
## 5

filter_sizes=[3,4,5])
char_out_dim = 30 * 3  # 3 filtros * 30 canales cada uno

# Dimensión total de entrada
total_input_dim = emb_dim + char_out_dim


2.2 (NER con LLMs)

Para el finetuning de los modelos  BETO y XML-Roberta  sobre el dataset prostata, se debe
tomar como referencia el entrenamiento del modelo base de bert-base-cased  sobre el
dataset Conll2002 (rendimiento del modelo es del 75% a 6 épocas, usando 8GB de GPUen
## Colab.




Tenga en cuenta que para el afinamiento  de  BETO y XML-Roberta se deben implementar
los  siguientes pasos:

-  Crear  su token de autenticación del sitio web de Hugging Face para subir el mejor
modelo entrenado.
- Implementar la fase de preprocesamiento y alineamiento del tokenizador,  realizar el
padding o relleno de manera dinámica usando   la librería
DataCollatorWithPadding.
-  Implementar el modelo de clasificación usando la  librería
AutoModelForTokenClassification.
- Definir la estrategia de entrenamiento usando TrainingArguments, parametrizando
completamente el modelo. Probar batch sizes de 8, 16 y 32 con 6-10 épocas.
5 Implementar el Trainer del modelo, definiendo los conjuntos de entrenamiento y
validación  y ejecutando el entrenamiento de ambos  modelos. Finalmente, subir el modelo
a Hugging Face.


Punto 3. NER con Prompt Engineering y LLMs Generativos

## 6

En esta tarea, se debe diseñar  un prompt para cada uno de los siguientes modelos
generativos:  Mistral-7B-v0.2, Gemma, Llama3-2-3B-instruct y Deepseek.
El objetivo es
extraer una lista de entidades de un texto clínico referente al cáncer de próstata:


● Edad y sexo del paciente
## ● Biomarcadores
● Tipo de cáncer
## ● Cirugías
● Edad del paciente
## ● Diagnóstico
● Hallazgos de imágenes
● Exámenes realizados
● Antecedentes médicos
● Resultados de laboratorios
● Tratamiento y medicación.

Como un ejemplo,  revise el modelo Mistral-7B-v0.2, que reconoce entidades nombradas al
estilo de Conll2002. Importante tener en cuenta que, al usar
prompt engineering, no se
realiza afinamiento; el modelo se guía únicamente mediante instrucciones
textuales. Aunque nativamente algunos LLMs (como
Mistral-7B-v0.2
) incluyen
etiquetas de instrucción ([INST] ... [/INST]), los componentes que se deben
definir para la extracción son:

- Llamado al modelo,
- Cuantización del modelo para ejecutarlo en GPU de Colab,
- Texto o argumento del pipeline,
- API de conexión a Hugging Face,
## 5. Prompt,
- Texto de entrada,
- Chat template específico del LLM para realizar la extracción.
Los siguientes son los  chat template de los modelos

## 7



A continuación, el texto del cual se extraerán las entidades (deberían ser similares a
las del dataset de
prostata
extraídas con clasificadores BERT):


Paciente masculino de 72 años con antecedentes médicos de hipertensión arterial (HTA).
Actualmente presenta un tumor prostático bilateral confirmado mediante biopsia. El
diagnóstico histológico reveló un adenocarcinoma de próstata. El puntaje de Gleason
reportado fue 3+3, lo que indica un cáncer de bajo grado. El PSA más alto registrado en la
historia clínica fue de 9,9 ng/dL. La resonancia magnética no evidenció lesiones
metastásicas ni afectación de la cápsula prostática. Las vesículas seminales se observaron
normales en las imágenes por resonancia. No se detectaron adenopatías ni lesiones óseas
sospechosas. La gammagrafía ósea fue negativa para metástasis. El cuadro clínico sugiere
una neoplasia localizada de bajo riesgo.






## 8