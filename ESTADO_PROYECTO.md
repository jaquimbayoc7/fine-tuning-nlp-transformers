# Estado del Proyecto - Taller Final NLP

## 📚 Estructura del Taller

El taller consta de **3 puntos principales**:

1. **Punto 1**: Clasificación de texto con LLMs (TASS + Sarcasmo)
2. **Punto 2**: Reconocimiento de Entidades Nombradas (NER)
   - **2.1**: NER con BiLSTMs sobre CoNLL-2002
   - **2.2**: NER con Fine-tuning (BETO + XLM-R sobre Próstata)
3. **Punto 3**: NER con Prompt Engineering (LLMs generativos - OPCIONAL)

---

## ✅ Trabajo Completado (4.25/4.5 puntos - 94.4%)

### ✅ Punto 1: Clasificación de Texto (2.5 puntos)

#### 1.1 Fine-tuning TASS (0.75 pts) ✅
- ✅ **BETO**: F1 64.54%
- ✅ **XLM-RoBERTa**: F1 64.41%
- ✅ Entrenamiento con batch sizes [8, 16, 32] en Kaggle
- ✅ Modelos publicados en HuggingFace:
  - `Lucyan85/beto-tass-sentiment`
  - `Lucyan85/xlmr-tass-sentiment`
- 📁 **Archivos**: `resultados/fase1_tass/`, `notebooks/2_Entrenamiento_TASS.ipynb`

#### 1.2 Fine-tuning Sarcasmo (0.75 pts) ✅
- ✅ **BETO**: F1 99.56% ⭐
- ✅ **XLM-RoBERTa**: F1 93.91%
- ✅ Entrenamiento con batch sizes [8, 16, 32] en Kaggle
- ✅ Modelos publicados en HuggingFace:
  - `Lucyan85/beto-sarcasmo-sentiment`
  - `Lucyan85/xlmr-sarcasmo-sentiment`
- 📁 **Archivos**: `resultados/fase2_sarcasmo/`, `notebooks/4_Entrenamiento_Sarcasmo.ipynb`

#### 1.3 Prueba de modelos HuggingFace (0.5 pts) ✅
- ✅ Notebook unificado con todos los modelos
- ✅ Probado con ejemplos de TASS y Sarcasmo
- ✅ Comparación de desempeño BETO vs XLM-R
- 📁 **Archivos**: `notebooks/8_Prueba_Modelos_HuggingFace_TASS.ipynb`

#### 1.4 Análisis Comparativo (0.5 pts) ✅
- ✅ Análisis: ¿Por qué TASS tiene menor rendimiento que Sarcasmo?
- ✅ Impacto del número de clases (2 vs 3)
- ✅ Comparación entre arquitecturas (BETO vs XLM-R)
- ✅ 7 archivos de análisis generados (gráficos + informes)
- 📁 **Archivos**: `resultados/fase3_comparativo/`, `notebooks/5_Analisis_Comparativo.ipynb`

---

### ✅ Punto 2.1: NER con BiLSTM (0.75 pts) ✅
- ✅ **Modelo 1**: BiLSTM + CRF + FastText → F1 64.21%
- ✅ **Modelo 2**: BiLSTM + CRF + CNN + FastText → F1 72.00% ⭐
- ✅ Mejora de **+7.70 puntos** sobre baseline (64.3%)
- ✅ Dataset: CoNLL-2002 (NER español)
- 📁 **Archivos**: `resultados/fase4_ner_bilstm/`, `notebooks/6_Exploracion_NER_CoNLL2002.ipynb`

---

### ✅ Punto 2.2: NER con Fine-tuning en Próstata (1.0 pts) ✅

#### Fine-tuning BETO + XLM-RoBERTa (0.75 pts) ✅
- ✅ **BETO**: F1 96.18% (Precision: 95.86%, Recall: 96.49%) ⭐
- ✅ **XLM-RoBERTa**: F1 96.70% (Precision: 96.41%, Recall: 97.00%) ⭐⭐
- ✅ Entrenamiento con batch_size=16, 10 épocas en Kaggle
- ✅ Mejora de **+24.70 puntos** sobre BiLSTM baseline (72.00%)
- ✅ Mejora de **+24.18 puntos** sobre BiLSTM-CoNLL (72.00%)
- ✅ Dataset: Próstata (10 tipos de entidades clínicas)
- 📁 **Archivos**: `resultados/fase5_ner_prostata/`, `notebooks/10_Entrenamiento_BETO_Prostata_Kaggle.ipynb`, `notebooks/11_Entrenamiento_XLM_R_Prostata_Kaggle.ipynb`

#### Publicación en HuggingFace (0.25 pts) ✅
- ✅ Modelos publicados en HuggingFace:
  - `Lucyan85/beto-ner-prostata`
  - `Lucyan85/xlmr-ner-prostata`
- ✅ Model cards completos con métricas y ejemplos
- ✅ Modelos probados con texto clínico de próstata
- ✅ Acceso público verificado

**Entidades reconocidas** (10 tipos):
- EDAD, BIOMARCADOR (PSA), CANCER, GLEASON, TNM
- TRATAMIENTO, MEDICAMENTO, DOSIS, CIRUGIA, FECHA

---

## 🔄 Trabajo Pendiente (0.25 puntos - 5.6%)

---

### 🎁 Punto 3: NER con Prompt Engineering (OPCIONAL)

**Objetivo**: Usar LLMs generativos con prompts (sin fine-tuning) para extraer entidades clínicas

**Modelos a probar**:
- [ ] Mistral-7B-v0.2
- [ ] Gemma
- [ ] Llama3-2-3B-Instruct
- [ ] DeepSeek

**Entidades a extraer del texto clínico**:
- Edad y sexo del paciente
- Biomarcadores (PSA, Gleason)
- Tipo de cáncer
- Cirugías y tratamientos
- Diagnóstico
- Hallazgos de imágenes
- Exámenes realizados
- Antecedentes médicos
- Resultados de laboratorio
- Medicación

**Componentes necesarios**:
- Cuantización del modelo (para ejecutar en Colab GPU)
- Chat template específico de cada LLM
- Diseño de prompt con instrucciones claras
- Pipeline de generación de texto
- Comparación vs fine-tuning (Punto 2.2)

**Prioridad**: 🟡 **BAJA** (opcional - no aparece en rúbrica con puntaje específico)

---

## 📊 Resumen de Puntaje

| Sección | Puntos | Estado |
|---------|--------|--------|
| Punto 1: Clasificación TASS + Sarcasmo | 2.5 / 2.5 | ✅ 100% |
| Punto 2.1: BiLSTM NER | 0.75 / 0.75 | ✅ 100% |
| Punto 2.2: Fine-tuning Próstata | 1.0 / 1.0 | ✅ 100% |
| Punto 3: Prompt Engineering | - / - | 🎁 Opcional |
| **TOTAL** | **4.25 / 4.5** | **94.4%** |

---

## 🎯 Próximos Pasos

1. **Inmediato**: Comenzar Punto 2.2 (Fine-tuning BETO + XLM-R en Próstata)
   - Crear notebook de exploración del dataset
   - Implementar preprocesamiento y alineamiento
   - Entrenar ambos modelos con diferentes batch sizes

2. **Después**: Publicar modelos en HuggingFace (0.25 pts)
   - Subir mejores modelos
   - Crear notebook de prueba

3. **Opcional**: Explorar Prompt Engineering (Punto 3)
   - Solo si hay tiempo disponible
   - Comparar efectividad vs fine-tuning
