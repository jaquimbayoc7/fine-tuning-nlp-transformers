# Templates de Prompts para NER con Prompt Engineering
# Punto 3: NER Biomédico con LLMs Generativos

"""
Este archivo contiene los templates de prompts para extraer entidades clínicas
de textos sobre cáncer de próstata usando diferentes estrategias de prompting.
"""

# ====================================================================
# VERSIÓN 1: ZERO-SHOT (Simple y directo)
# ====================================================================

PROMPT_ZERO_SHOT = """Eres un asistente médico experto en oncología especializado en cáncer de próstata.

Tu tarea es extraer información clínica estructurada del siguiente texto médico.

ENTIDADES A EXTRAER:
1. edad: Edad del paciente (ej: "72 años", "65 años")
2. sexo: Sexo del paciente (ej: "masculino", "femenino")
3. biomarcadores: PSA, Gleason, TNM, marcadores tumorales
4. tipo_cancer: Tipo específico de cáncer o neoplasia
5. cirugias: Procedimientos quirúrgicos realizados
6. diagnostico: Diagnóstico principal o hallazgos diagnósticos
7. imagenes: Hallazgos de estudios por imagen (resonancia, TAC, gammagrafía)
8. examenes: Exámenes y pruebas realizadas
9. antecedentes: Antecedentes médicos del paciente
10. tratamiento: Tratamientos aplicados
11. medicacion: Medicamentos prescritos o dosis

FORMATO DE SALIDA:
Responde ÚNICAMENTE con un objeto JSON válido. NO agregues explicaciones, comentarios ni texto adicional.
Si una categoría no tiene información, usa una lista vacía [].

TEXTO A ANALIZAR:
{texto}

JSON:"""

# ====================================================================
# VERSIÓN 2: FEW-SHOT (Con ejemplos)
# ====================================================================

PROMPT_FEW_SHOT = """Eres un asistente médico experto en oncología. Tu tarea es extraer entidades clínicas de textos médicos.

INSTRUCCIONES:
- Extrae información estructurada en formato JSON
- Identifica cada tipo de entidad mencionada
- Usa listas para agrupar múltiples valores del mismo tipo
- Si no encuentras información para una categoría, usa lista vacía []

CATEGORÍAS A EXTRAER:
edad, sexo, biomarcadores, tipo_cancer, cirugias, diagnostico, imagenes, examenes, antecedentes, tratamiento, medicacion

---

EJEMPLO 1:

Texto: "Paciente femenino de 65 años con antecedentes de diabetes. Diagnóstico de cáncer de mama estadio II. TAC de tórax sin metástasis. Tratamiento con quimioterapia adyuvante."

JSON:
{{
  "edad": ["65 años"],
  "sexo": ["femenino"],
  "biomarcadores": [],
  "tipo_cancer": ["cáncer de mama estadio II"],
  "cirugias": [],
  "diagnostico": ["cáncer de mama estadio II"],
  "imagenes": ["TAC de tórax sin metástasis"],
  "examenes": ["TAC de tórax"],
  "antecedentes": ["diabetes"],
  "tratamiento": ["quimioterapia adyuvante"],
  "medicacion": []
}}

---

EJEMPLO 2:

Texto: "Varón de 58 años. PSA de 12.5 ng/dL. Gleason 4+3. Biopsia confirma adenocarcinoma prostático. Resonancia magnética muestra lesión localizada. Sin invasión capsular."

JSON:
{{
  "edad": ["58 años"],
  "sexo": ["masculino"],
  "biomarcadores": ["PSA 12.5 ng/dL", "Gleason 4+3"],
  "tipo_cancer": ["adenocarcinoma prostático"],
  "cirugias": ["biopsia"],
  "diagnostico": ["adenocarcinoma prostático"],
  "imagenes": ["resonancia magnética con lesión localizada", "sin invasión capsular"],
  "examenes": ["biopsia", "resonancia magnética"],
  "antecedentes": [],
  "tratamiento": [],
  "medicacion": []
}}

---

AHORA ES TU TURNO:

Texto: {texto}

JSON:"""

# ====================================================================
# VERSIÓN 3: CHAIN-OF-THOUGHT (Razonamiento paso a paso)
# ====================================================================

PROMPT_CHAIN_OF_THOUGHT = """Eres un asistente médico experto en oncología. Vas a extraer información clínica de un texto médico de forma sistemática.

PROCESO PASO A PASO:

1. LEE CUIDADOSAMENTE: Lee el texto completo identificando frases clave
2. IDENTIFICA ENTIDADES: Busca sistemáticamente cada tipo de entidad
3. EXTRAE TEXTO EXACTO: Copia el texto exacto de cada entidad encontrada
4. ORGANIZA EN JSON: Estructura la información en formato JSON

CATEGORÍAS DE ENTIDADES:
- edad: Edad del paciente
- sexo: Sexo del paciente  
- biomarcadores: PSA, Gleason, TNM, valores de laboratorio
- tipo_cancer: Tipo específico de cáncer o tumor
- cirugias: Procedimientos quirúrgicos
- diagnostico: Diagnóstico o hallazgos diagnósticos
- imagenes: Resultados de estudios por imagen
- examenes: Pruebas y exámenes realizados
- antecedentes: Historia médica previa
- tratamiento: Terapias aplicadas
- medicacion: Fármacos y dosis

---

TEXTO A ANALIZAR:
{texto}

---

PIENSA PASO A PASO:

Paso 1 - Lectura inicial:
[Identifica los conceptos principales del texto]

Paso 2 - Extracción por categoría:
[Busca sistemáticamente cada tipo de entidad]

Paso 3 - Verificación:
[Confirma que no falta información relevante]

Paso 4 - Formato JSON:
[Estructura final en JSON válido]

JSON FINAL:"""

# ====================================================================
# VERSIÓN 4: INSTRUCTIVO DETALLADO (Formato estricto)
# ====================================================================

PROMPT_INSTRUCTIVO = """## INSTRUCCIONES DE EXTRACCIÓN DE ENTIDADES CLÍNICAS

**ROL**: Eres un sistema experto en procesamiento de informes médicos de oncología.

**TAREA**: Extraer entidades clínicas de un texto sobre cáncer de próstata.

**REGLAS ESTRICTAS**:
1. Responde SOLO con JSON válido
2. NO agregues comentarios fuera del JSON
3. Usa el texto EXACTO del documento original
4. Si una categoría no tiene datos, usa []
5. Mantén el formato de las listas consistente

**ESQUEMA JSON**:
```json
{{
  "edad": ["string"],
  "sexo": ["string"],
  "biomarcadores": ["string", "string", ...],
  "tipo_cancer": ["string", "string", ...],
  "cirugias": ["string", "string", ...],
  "diagnostico": ["string", "string", ...],
  "imagenes": ["string", "string", ...],
  "examenes": ["string", "string", ...],
  "antecedentes": ["string", "string", ...],
  "tratamiento": ["string", "string", ...],
  "medicacion": ["string", "string", ...]
}}
```

**DEFINICIONES**:
- **edad**: Edad numérica con unidad (ej: "72 años")
- **sexo**: "masculino" o "femenino"
- **biomarcadores**: PSA (ng/dL o ng/mL), Gleason (ej: "3+3", "4+3"), TNM
- **tipo_cancer**: Nombre del cáncer (ej: "adenocarcinoma de próstata")
- **cirugias**: Procedimientos quirúrgicos (biopsia, prostatectomía, etc.)
- **diagnostico**: Hallazgos diagnósticos y conclusiones
- **imagenes**: Resultados de resonancia, TAC, gammagrafía, ecografía
- **examenes**: Pruebas realizadas (laboratorio, imagen, patología)
- **antecedentes**: Historia médica previa (HTA, diabetes, etc.)
- **tratamiento**: Terapias aplicadas (radioterapia, quimioterapia, etc.)
- **medicacion**: Fármacos específicos y sus dosis

---

**TEXTO MÉDICO**:
{texto}

---

**SALIDA JSON**:"""

# ====================================================================
# MAPEO DE VERSIONES
# ====================================================================

PROMPTS = {
    'v1': PROMPT_ZERO_SHOT,
    'v2': PROMPT_FEW_SHOT,
    'v3': PROMPT_CHAIN_OF_THOUGHT,
    'v4': PROMPT_INSTRUCTIVO
}

# ====================================================================
# FUNCIÓN DE AYUDA
# ====================================================================

def get_prompt(version='v1'):
    """
    Obtiene el template de prompt según la versión.
    
    Args:
        version (str): 'v1', 'v2', 'v3', o 'v4'
    
    Returns:
        str: Template del prompt
    """
    return PROMPTS.get(version, PROMPT_ZERO_SHOT)

def format_prompt(texto, version='v1'):
    """
    Formatea el prompt con el texto médico.
    
    Args:
        texto (str): Texto clínico a analizar
        version (str): Versión del prompt a usar
    
    Returns:
        str: Prompt completo listo para enviar al LLM
    """
    template = get_prompt(version)
    return template.format(texto=texto)

# ====================================================================
# METADATA
# ====================================================================

PROMPT_METADATA = {
    'v1': {
        'nombre': 'Zero-shot',
        'descripcion': 'Instrucción simple y directa sin ejemplos',
        'complejidad': 'baja',
        'tokens_aproximados': 200
    },
    'v2': {
        'nombre': 'Few-shot',
        'descripcion': 'Incluye 2 ejemplos anotados',
        'complejidad': 'media',
        'tokens_aproximados': 500
    },
    'v3': {
        'nombre': 'Chain-of-thought',
        'descripcion': 'Guía el razonamiento paso a paso',
        'complejidad': 'media',
        'tokens_aproximados': 350
    },
    'v4': {
        'nombre': 'Instructivo detallado',
        'descripcion': 'Especificaciones detalladas y formato estricto',
        'complejidad': 'alta',
        'tokens_aproximados': 400
    }
}
