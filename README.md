# Diagnóstico de marca empleadora de Tesla con NLP (Glassdoor)

Análisis de reseñas de empleados en Glassdoor para identificar las fortalezas y debilidades de **Tesla** como empleador frente a sus competidores del sector automovilístico: **Ford, General Motors, Stellantis y Honda**.

Proyecto del módulo de Deep Learning / NLP del Máster en Data Science & IA (Evolve).

## Objetivo

Responder a una pregunta de Recursos Humanos: *¿qué temas generan más quejas entre los empleados de Tesla y en cuáles está peor que el sector?* El resultado es una lista priorizada de áreas de mejora, pensada para un público no técnico.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `NLP_Glassdoor_FernandoLopez.ipynb` | Notebook con todo el análisis: limpieza, sentimiento, topic modeling, comparación con peers y conclusiones |
| `Presentacion_Tesla_Glassdoor.pptx` | Presentación de resultados para el equipo de RR. HH. |
| `glassdoor_reviews_hr.parquet` | Dataset de reseñas de Glassdoor (almacenado con Git LFS) |
| `requirements.txt` | Dependencias de Python |

## Metodología

1. **Limpieza y EDA.** Filtrado de Tesla y peers, revisión de nulos y duplicados. Se documentan dos problemas de Glassdoor: General Motors aparece con una página duplicada, y la página de Stellantis solo cubre reseñas desde 2021, por lo que se une con el histórico de FCA (Stellantis nace de la fusión FCA + PSA en enero de 2021).
2. **Análisis de sentimiento** con DistilBERT (SST-2) sobre los campos *pros* y *cons*, como control de calidad del texto.
3. **Topic modeling con BERTopic** (embeddings `all-MiniLM-L6-v2`, UMAP + HDBSCAN). Los temas se agrupan en 11 categorías de negocio (compensación, conciliación, calidad del management, etc.).
4. **Comparación con peers** mediante un *balance neto* por categoría: % de menciones positivas − % de menciones negativas.
5. **Empleados actuales vs. antiguos**, para aproximar los motivos de salida.
6. **Matriz de priorización**, que cruza el volumen de quejas con la desventaja frente a los peers.

## Principales resultados

- La valoración media de Tesla (**3,59**) está por debajo de la de sus peers (**3,78**).
- Las principales debilidades son la **conciliación / horarios** y la **calidad del management**.
- **Compensación** y **estabilidad / burocracia** están en la media del sector o por encima.

El detalle, las recomendaciones y las limitaciones están en la sección final del notebook y en la presentación.

## Cómo ejecutarlo

### 1. Clonar el repositorio

El parquet se guarda con [Git LFS](https://git-lfs.com), así que hay que instalarlo antes de clonar:

```bash
git lfs install
git clone https://github.com/FLopez26/<nombre-del-repo>.git
cd <nombre-del-repo>
```

### 2. Crear el entorno (Python 3.12)

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate        # Git Bash  (PowerShell: .\.venv\Scripts\Activate.ps1)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para usar GPU NVIDIA, reinstala `torch` con soporte CUDA:

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### 3. Abrir el notebook

```bash
jupyter notebook NLP_Glassdoor_FernandoLopez.ipynb
```

**Nota:** la primera ejecución del sentimiento y de los embeddings es lenta en CPU. El notebook guarda los resultados en caché (`cache_sent_*.parquet`, `cache_emb_*.npy`) en la misma carpeta, de modo que las ejecuciones siguientes son mucho más rápidas. Estas cachés no se incluyen en el repositorio.

## Limitaciones

- Las reseñas de Glassdoor son voluntarias y tienen sesgo de autoselección.
- La asignación de temas a categorías de negocio incluye criterio manual.
- El modelo de sentimiento está entrenado en inglés y en un dominio distinto (reseñas de cine).

## Autor

**Fernando López Sánchez**: [github.com/FLopez26](https://github.com/FLopez26)
