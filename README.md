# **CLASIFICADOR DE FAKE-NEWS ACERCA DEL COVID-19**

## **1.-Descripción del problema**

Durante la pandemia de COVID-19, la propagación del virus vino acompañada de una enorme cantidad de noticias que se difundían en cuestión de minutos por todo el mundo. Noticias falsas sobre curas milagrosas, efectos secundarios de las candidatas a vacunas que se inventaban o teorías conspirativas acerca de la falsedad de la enfermedad circularon ampliamente, afectando a la percepción pública y mezclando información valiosa con bulos que circulaban por la red.

En este proyecto se han desarrollado diferentes modelos de clasificación automáticos capaces de distinguir entre noticias verdaderas y falsas relacionadas con el Covid-19 mediante técnicas de Procesamiento del Lenguaje Natural (NLP). El reto pasa por conseguir detectar patrones y diferenciar contenido informativo real de contenido engañoso y manipulador.

## **2.-Conjunto de datos**

Para abordar el problema, hemos contado con una base de datos con noticias recopiladas de diferentes fuentes y etiquetadas como 1-True o 0-False/Partially False. Concretamente, hay un total de 3119 noticias, repartidas entre 2061 verdaderas y 1058 falsas (como muestra el histograma a continuación), de las que no solo se muestra la noticia y la etiqueta, sino también el título y la categoría (True/False/Partially False). 

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/f19ed717-adc3-4503-b334-b4a20e5a54a9" />

En cuanto a esta subdivisión de las noticias falsas entre False y Partially False, dado que el número de noticias verdaderas es casi el doble, hemos podido comprobar que provoca una degradación en las prestaciones de algunos de los algoritmos de clasificación al intentar clasificar entre 3 etiquetas diferentes. Es por ello que se ha simplificado el problema a un clasificador binario de noticias true/false sobre el COVID-19 eliminando la categoría de Partially False.

Las noticias de nuestro dataset tienen de media unos 2250 caracteres y 365 palabras, es decir, que son noticias bastante largas. En el histograma de la siguiente figura se muestra la distribución del número de caracteres y palabras y la frecuencia con la que aparecen en las noticias. Encontramos noticias de hasta 5500 palabras y 32000 caracteres, luego para futuras secciones, aunque se reducirá considerablemente el vocabulario empleado, nos quedaremos con unas 8000 palabras. Es algo más de lo habitual, pero el objetivo es no perder información en nuestras noticias tan grandes.

<img width="1624" height="573" alt="image" src="https://github.com/user-attachments/assets/e0ed4707-3508-4b86-95c3-74e5e58e23ca" />

Como hipótesis y resultados que podríamos esperar, podemos decir que, al haber casi el doble de noticias verdaderas que de noticias falsas, es probable que sea para la clase maypritaria (verdaderas) para la que cometamos menos errores al clasificar las noticias. Además, no hemos podido identificar en el primer cuaderno (de análisis de la base de datos) ninguna palabra característica o patrón importante que diferencie con claridad las noticias verdaderas y falsas.

Como trabajo previo a la presentación de las técnicas de vectorización y algoritmos clasificadores, se ha realizado una limpieza y homogeneización de este conjunto de noticias para evitar que caracteres, urls y palabras concretas (como cambiar *viru* por *virus*) entren a nuestros modelos e introduzcan ruido que dificulte el trabajo de clasificación.

## **3.-Metodologías utilizadas**

### **3.1. Técnicas de vectorización:**

Se ha presentado en el punto anterior el dataset (cuaderno 1 en github) con noticias desde el que partimos para entrenar, validar y testear nuestros modelos. Sin embargo, entrando ya en el cuaderno 2 subido a github, para entrenar un algoritmo de clasificación como SVM, redes neuronales, etc. necesitamos convertir el texto en representaciones numéricas, para lo cual empleamos 3 técnicas de vectorización diferentes que se presentan a continuación. Destacar que en este apartado del proyecto, tán solo hemos definido y experimentado con estos tres modelos aplicados sobre la base de datos global y será en el siguiente apartado donde dividiremos el dataset en *train*,*test* y *validation*.

- ***TF-IDF:*** Esta técnica transforma cada una de nuestras noticias en un vector basándose únicamente en la importancia de las palabras en las noticias. Para implementarlo, se han propuesto 2 alternativas con dos librerías de python: gensim (partiendo de una descripción *Bag of Words*) y un modelo ya definido de sklearn. Las prestaciones en ambos casos hemos comprobado que son prácticamente iguales y por simplicidad, se ha utilizado el modelo de sklearn, que internamente realiza la tokenización de las noticias, vectorización y reducción del vocabulario. Esto es un punto importante, pues el vocabulario original de la base de datos está en torno a 37000 palabras, lo cual es difícil (e innecesario) de manejar por los modelos. Es por ello que se han suprimido aquellas palabras que han aparecido en más del 60% de las noticias o que aparecen en menos de 10.
  
- ***Word2Vec:*** Es un modelo que aprende embeddings semánticos de palabras, es decir, vectores densos donde palabras con significados similares tienen representaciones similares. Es una técnica a priori más compleja que TF-IDF (que solo se basa en frecuencia de aparición de las palabras) y esto se ve reflejado en los resultados que se presentarán posteriormente. A diferencia de TF-IDF, que partía de las noticias "limpias y homogeneizadas", en este caso se parte de los tokens de dichas noticias. Igual que antes, se ha realizado una reducción del vocabulario antes de calcular embeddings y para facilitar tareas futuras.

- ***Embeddings contextuales:*** Son vectores generados por modelos donde la representación de una palabra depende del contexto en el que aparece, es decir, capturan relaciones complejas entre palabras y son adecuados para representaciones profundas. En nuestro caso, igual que con TF-IDF, se ha realizado una comparativa entre 3 posibles modelos: BERT, RoBERTa y distilBERT. En este caso, para la elección del modelo más adecuado, hemos realizado una pequeña comparativa con una porción de la base de datos para ver cuál es el que ofrece mejores prestaciones. El resultado es que, para nuestra aplicación de clasificador *true*/*false*, es el modelo de RoBERTa el que mejor funciona, luego ha sido el que se ha escogido para los clasificadores que se presentan en el siguiente punto.

### **3.2. Modelos de clasificación:**

Las técnicas de vectorización planteadas se han utilizado para adecuar las noticias de nuestra base de datos a entradas válidas que sirvan para 3 modelos que se han evaluado: 2 algoritmos de Scikit-Learn (Regresión Logística y SVM) y una red neuronal que hemos diseñado. Por otra parte, se ha utilizado fine-tunning sobre el modelo de RoBERTa hallado en el anterior punto (aunque aquí no se utilizan vectorizaciones). En cada uno de los casos, se han ido utilizando diferentes métricas de evaluación de resultados destacando *accuracy* para Regresión Logística y SVM y *matriz de confusión*, *f1-score* y *recall* para la red neuronal y modelo con fine-tunning.

Además, se ha dividido la base de datos en 3 conjuntos de *train*,*test* y *validation*, con una cantidad de noticias del 60%, 20% y 20% del total respectivamente. Así, tanto para la red neuronal como para Regresión Logística y SVM, se han utilizado los modelos definidos en el punto anterior aplicados sobre los tres conjuntos, de forma que cada técnica de vectorización tenga su dataset de *train*,*test* y *validation*.

- ***Regresión Logística:*** Es un modelo estadístico lineal que predice la probabilidad de que una instancia pertenezca a una clase usando una función sigmoide. A priori es el clasificador más simple de los utilizados, pero como veremos en el siguiente punto, las prestaciones que ofrece son realmente buenas.

- ***Supporrt Vector Machine (SVM):*** Es un modelo de aprendizaje supervisado que busca un límite óptimo que separe las clases con el mayor margen posible (en este caso nuestras noticias *true*/*false*). Se ha escogido por ser un modelo bastante diferente del de Regresión Logística y, en principio, más complejos. En el siguiente apartado se ofrecen los resultados obtenidos en comparación con Regresión Logística.
  
- ***Red Neuronal:*** En este caso se ha diseñado un modelo supervisado compuesto por capas de nodos (neuronas) para clasificar noticias sobre COVID-19 como verdaderas o falsas usando vectores TF-IDF, Word2Vec y Embeddings Contextuales (RoBERTa).

  Hemos implementado una arquitectura con tres capas ocultas (de 1024, 512 y 128 neuronas), con ReLU, Batch Normalization, Dropout progresivo para prevenir sobreajuste y una capa de salida con sigmoide para clasificación binaria. Dado que es binaria, se ha establecido la red con una única salida para reducir la complejidad de la misma, pues es suficiente para deducir si la noticia es verdadera (salida=1) o falsa (salida=0).

  Para el entrenamiento, hemos utilizado un optimizador Adam y BCEWithLogitsLoss como función de pérdidas, que combina sigmoide + Binary Cross Entropy. Los datos se procesan en mini-batches con validación para controlar el rendimiento y se han ido ajustando los parámetros de learning rate, número de neuronas de las capas ocultas, dropout, weight decay, etc. hasta obtener los resultados que se presentan en la siguiente sección.
  
- ***RoBERTa+Fine-Tunning:*** Se ha realizado fine-tunning al modelo preentrenado de RoBERTa. El fine-tuning consiste en ajustar un modelo preentrenado a una tarea específica usando tus propios datos. En nuestro caso, hemos entrenado RoBERTa con noticias sobre COVID-19 para que aprenda a clasificar automáticamente noticias verdaderas y falsas, combinando su conocimiento general del lenguaje con los patrones específicos de nuestro dataset. Los resultados han sido positivos y, nuevamente, se presentan y comparan en la siguiente sección.
  
- ***TRABAJO DE EXTENSIÓN:*** Como trabajo extra para complementar los resultados obtenidos, se ha realizado un análisis del sentimiento (positivo, negativo o neutro) para cada una de las noticias. el objetivo es verificar si la hipótesis de la que partíamos, que en las noticias falsas suele predominar un sentimiento negativo (dentro de que durante el COVID-19, la mayor parte de las noticias eran negativas) es verdad o no. Para ello, se han realizado 2 análisis con 2 librerías diferentes: VADER y Flair. Con la primera se han analizado los títulos de las noticias, que hasta este momento no se habían utilizado, clasificando el sentimiento como positivo, negativo o neutro. Con Flair, se han categorizado las noticias analizando el cuerpo de las mismas y suprimiendo la categoría neutra. El motivo por el que se han utilizado librerías diferentes es que con VADER, para textos muy largos (como el caso de nuestras noticias), el sentimiento suele caer en neutro y por lo tanto no íbamos a poder extraer demasiadas conclusiones. Esta información nos permite explorar patrones adicionales que podrían ayudar a distinguir noticias verdaderas de falsas.

## **4.-Resultados experimentales**
### **4.1. Protocolo experimental y métricas de evaluación**

Para la evaluación de los distintos modelos de clasificación propuestos, se han usado métricas que permiten comparar de forma justa las diferentes combinaciones de técnicas de vectorización y algoritmos de clasificación. 
En primer lugar, el conjunto de datos se ha dividido en tres subconjuntos: un 60% para entrenamiento, un 20% para validación y un 20% para test, manteniendo la proporción de clases en cada uno de ellos. El conjunto de entrenamiento se ha utilizado para el ajuste de los parámetros de los modelos, el de validación para monitorizar el rendimiento de los modelos durante el entrenamiento y comprobar su capacidad de generalización y el de test para la evaluación final del rendimiento.
Como ya hemos visto el problema se ha abordado como una tarea de clasificación binaria, donde cada noticia se etiqueta como true o false. Esta simplificación nos ha permitido centrar el análisis en la capacidad de los modelos para distinguir información veraz de información falsa, evitando la degradación de prestaciones observada en experimentos preliminares con clasificación multiclase (true, false o partially false).
Para la evaluación se han utilizado distintos indicadores en función del tipo de modelo evaluado. Para los modelos clásicos de Regresión Logística y Support Vector Machine (SVM) se ha empleado principalmente la accuracy, ya que, tras la limpieza y balanceo del dataset, esta métrica resulta representativa del rendimiento global del clasificador. No obstante, para los modelos más complejos, como la red neuronal y el modelo RoBERTa con fine-tuning, se han considerado métricas adicionales como la matriz de confusión, el recall y el F1-score, con especial atención a la capacidad del modelo para detectar correctamente noticias falsas.

### **4.2. Análisis y resultados de las técnicas de vectorización**

Antes de evaluar el rendimiento de los distintos modelos de clasificación, se ha realizado un análisis de las representaciones numéricas generadas por cada una de las técnicas de vectorización empleadas. Estas representaciones constituyen la entrada de los clasificadores y, por tanto, condicionan de forma directa su capacidad para extraer patrones relevantes del texto. 

#### **4.2.1. TF-IDF**

En una primera aproximación se ha utilizado la representación TF-IDF partiendo de un modelo Bag of Words. Inicialmente, el vocabulario generado alcanza un tamaño cercano a 37 000 palabras, dando lugar a una matriz de documentos de dimensiones (3119, 37385), lo que resulta poco manejable para los modelos de clasificación.
Para reducir la dimensionalidad, se ha empleado la implementación de scikit-learn, aplicando filtros sobre el vocabulario mediante los parámetros min_df y max_df. En concreto, se han eliminado las palabras que aparecen en menos de 10 documentos y aquellas que aparecen en más del 60% de las noticias. Tras este filtrado, el vocabulario se reduce a 7158 términos, obteniéndose una matriz final de tamaño (3119, 7158), sin pérdida apreciable de información relevante para la clasificación.

#### **4.2.2. Word2Vec**

Como segunda técnica se ha utilizado Word2Vec, que genera embeddings semánticos densos para cada palabra. Tras el entrenamiento del modelo, se obtiene un vocabulario de 8472 palabras, cada una representada mediante un vector de 100 dimensiones.
La representación de cada noticia se ha calculado como el promedio de los embeddings de las palabras que la componen, dando lugar a una matriz final de documentos de dimensiones (3119, 100). Esta reducción de dimensionalidad disminuye el coste computacional, aunque puede implicar una pérdida de información respecto a técnicas basadas en frecuencia como TF-IDF.

#### **4.2.3. Embeddings contextuales (BERT, DistilBERT y RoBERTa)**

Finalmente, se han evaluado embeddings contextuales basados en modelos preentrenados de lenguaje: DistilBERT, BERT y RoBERTa. Para una primera comparativa, se ha utilizado un subconjunto del dataset, obteniendo embeddings de 768 dimensiones y evaluando la accuracy en validación.
Los resultados muestran que RoBERTa ofrece el mejor rendimiento entre los modelos evaluados, por lo que se ha seleccionado como modelo de referencia. Aplicado sobre el conjunto completo de noticias, se obtiene una matriz final de documentos de dimensiones (3119, 768), utilizada en los experimentos posteriores.

### **4.3. Resultados con Regresión Logística**

En este apartado se presentan los resultados obtenidos al aplicar Regresión Logística sobre las distintas representaciones vectoriales analizadas previamente: TF-IDF, Word2Vec y embeddings contextuales basados en RoBERTa. Para cada caso se evalúa el rendimiento sobre los conjuntos de entrenamiento, validación y test mediante la métrica de accuracy.Los resultados obtenidos quedan resumidos en la siguiente tabla:

| Vectorización | Accuracy (Test) | Error rate (Test) |
|:--------------|:---------------:|:-----------------:|
| TF-IDF        | 0.7933          | 0.2067            |
| Word2Vec      | 0.8045          | 0.1955            |
| RoBERTa       | 0.7660          | 0.2340            |

En el caso de TF-IDF, la Regresión Logística alcanza los mejores resultados globales, con una accuracy del 87.7% en entrenamiento, 78.0% en validación y 79.3% en test. La diferencia moderada entre entrenamiento y test indica una buena capacidad de generalización, confirmando que esta representación basada en frecuencia resulta altamente efectiva para el problema planteado.
Para Word2Vec, los resultados obtenidos son algo inferiores en entrenamiento (80.9%), pero comparables e incluso ligeramente superiores en validación (80.1%) y test (80.5%). Este comportamiento sugiere que la representación semántica densa permite una generalización más estable, aunque con menor capacidad de ajuste que TF-IDF.
Por último, al emplear embeddings contextuales de RoBERTa junto con Regresión Logística, se obtienen valores de accuracy del 80.1% en entrenamiento, 77.6% en validación y 76.6% en test. A pesar de la mayor complejidad de esta representación, el clasificador lineal no consigue explotar plenamente la información contextual capturada por los embeddings, obteniendo resultados ligeramente inferiores a los de TF-IDF y Word2Vec.
Estos resultados muestran que la Regresión Logística, a pesar de su simplicidad, ofrece un rendimiento muy competitivo cuando se combina con representaciones adecuadas, destacando especialmente la combinación con TF-IDF como una sólida línea base para el problema de clasificación de noticias true/false. En cambio, no explota todas las capacidades de técnicas más complejas como Word2Vec y RoBERTa.

### **4.4. Resultados con Support Vector Machine (SVM)**

En este apartado se presentan los resultados obtenidos al aplicar un clasificador SVM lineal sobre las distintas representaciones vectoriales analizadas: TF-IDF, Word2Vec y embeddings contextuales basados en RoBERTa. Al igual que en el caso anterior, el rendimiento se evalúa sobre los conjuntos de entrenamiento, validación y test utilizando la métrica de accuracy.
Los resultados obtenidos en el conjunto de test se resumen en la siguiente tabla:

| Vectorización | Accuracy (Test) | Error rate (Test) |
|:--------------|:---------------:|:-----------------:|
| TF-IDF        | 0.7901          | 0.2099            |
| Word2Vec      | 0.8029          | 0.1971            |
| RoBERTa       | 0.7580          | 0.2420            |

En el caso de TF-IDF, el clasificador SVM lineal alcanza una accuracy del 92.4% en entrenamiento, 78.7% en validación y 79.0% en test. La diferencia entre entrenamiento y test sugiere una buena capacidad de ajuste del modelo, aunque con una ligera pérdida de generalización respecto al conjunto de entrenamiento.
Para la representación basada en Word2Vec, se obtienen valores de accuracy del 81.6% en entrenamiento, 80.5% en validación y 80.3% en test. Estos resultados muestran un comportamiento más equilibrado entre los distintos conjuntos, indicando una mayor estabilidad del modelo cuando se utilizan embeddings semánticos densos.
Por último, al emplear embeddings contextuales de RoBERTa junto con SVM lineal, se obtiene una accuracy del 81.9% en entrenamiento, 78.5% en validación y 75.8% en test. A pesar de la riqueza de la representación, el modelo no logra mejorar los resultados obtenidos con otras técnicas, lo que sugiere que el clasificador lineal no es capaz de explotar completamente la información contextual capturada por los embeddings.
Los resultados muestran que el SVM lineal presenta un comportamiento consistente en función de la representación utilizada, siendo especialmente estable cuando se combina con Word2Vec, mientras que su rendimiento disminuye al emplear embeddings contextuales de mayor complejidad como RoBERTa.

### **4.5 Comparativa Regresión Logística vs SVM**
Para terminar con la parte de clasificadores clásicos , en este apartado se va a mostrar una breve comparación entre Regresión Logística y  SVM. Para ello usaremos los datos de test resumidos en la siguiente imagen y tabla:

<img width="987" height="333" alt="image" src="https://github.com/user-attachments/assets/77e13069-9b93-466c-a045-a9082450fb52" />

| Vectorización | Accuracy (Logistic) | Accuracy (SVM) | Error rate (Logistic) | Error rate (SVM) |
|:--------------|:-------------------:|:--------------:|:---------------------:|:----------------:|
| TF-IDF        | 0.7933              | 0.7901         | 0.2067                | 0.2099           |
| Word2Vec      | 0.8045              | 0.8029         | 0.1955                | 0.1971           |
| RoBERTa       | 0.7660              | 0.7580         | 0.2340                | 0.2420           |

Como se aprecia en los valores mostrados, ambos clasificadores presentan comportamientos muy similares en el conjunto de test para cada técnica de vectorización. Las diferencias entre Regresión Logística y SVM lineal son reducidas, siendo Word2Vec la representación que ofrece los mejores resultados en términos de accuracy para ambos modelos, mientras que RoBERTa, combinada con clasificadores lineales, muestra un rendimiento inferior. Estas observaciones refuerzan la idea de que la elección de la representación vectorial tiene un impacto mayor que la elección entre estos dos clasificadores lineales.

### **4.6. Resultados con Red Neuronal**

En este apartado se analizan los resultados obtenidos con la red neuronal implementada en PyTorch, aplicada a las distintas representaciones vectoriales consideradas: TF-IDF, Word2Vec y embeddings contextuales basados en RoBERTa. Para cada caso se presentan las métricas obtenidas en el conjunto de test, así como el comportamiento observado durante el entrenamiento a partir de las curvas de pérdida y precisión.

#### **4.6.1. TF-IDF**

| Métrica (Test)    | Valor  |
| ----------------- | ------ |
| Accuracy          | 0.7949 |
| Precision (macro) | 0.7785 |
| Recall (macro)    | 0.7496 |
| F1-score (macro)  | 0.7600 |

Como se puede ver en la tabla, la red neuronal entrenada sobre representaciones TF-IDF alcanza un rendimiento competitivo, con una accuracy cercana al 80% en el conjunto de test. Las métricas de precision, recall y F1-score muestran un comportamiento equilibrado, lo que indica una capacidad razonable para discriminar entre noticias verdaderas y falsas.

Sin embargo, si nos fijamos en las curvas de entrenamiento y test, el modelo presenta una rápida reducción de la pérdida en entrenamiento acompañada de un incremento progresivo de la pérdida en test. Este comportamiento evidencia un sobreajuste temprano, lo que limita la capacidad de generalización del modelo cuando se utilizan representaciones de alta dimensionalidad como TF-IDF.

<img width="1087" height="443" alt="image" src="https://github.com/user-attachments/assets/7653859c-6620-4f3e-a107-d831dbbc0106" />

Si nos fijamos en la matriz de confusión recogida en la siguiente tabla:

| Real \ Predicha | 0 (False) | 1 (True) |
|-----------------|-----------|----------|
| 0 (False)       | 129       | 83       |
| 1 (True)        | 45        | 367      |

Vemos que TF-IDF muestra un mayor número de falsos positivos, es decir, noticias falsas clasificadas como verdaderas. Este desequilibrio indica que el modelo tiende a favorecer la clase mayoritaria, lo que reduce la precisión en la detección de noticias falsas. Por este motivo, métricas como el F1-score resultan especialmente relevantes para evaluar el rendimiento real del clasificador en este escenario.

#### **4.6.1. Word2Vec**

| Métrica (Test)    | Valor  |
| ----------------- | ------ |
| Accuracy          | 0.7901 |
| Precision (macro) | 0.7659 |
| Recall (macro)    | 0.7712 |
| F1-score (macro)  | 0.7684 |

Como se puede ver resumido en la tabla, al emplear Word2Vec como técnica de vectorización, la red neuronal obtiene resultados similares en términos de accuracy, aunque con una ligera mejora en métricas como recall y F1-score. Esto sugiere una mejor capacidad del modelo para detectar correctamente ambas clases, especialmente la clase minoritaria.

Además, las curvas de pérdida y precisión muestran un comportamiento más estable que en el caso de TF-IDF, con una menor divergencia entre entrenamiento y test durante las primeras épocas. No obstante, el modelo sigue mostrando signos de sobreajuste a medida que avanza el entrenamiento, lo que limita las ganancias obtenidas frente a modelos más simples.

<img width="1083" height="441" alt="image" src="https://github.com/user-attachments/assets/07a5a889-5665-4f1d-8b0b-cb769bbac5e5" />

Si nos fijamos en la matriz de confusión recogida en la siguiente tabla:

| Real \ Predicha | 0 (False) | 1 (True) |
|-----------------|-----------|----------|
| 0 (False)       | 151       | 61       |
| 1 (True)        | 70        | 342      |

Para Word2Vec, los errores de clasificación se encuentran más equilibrados entre falsos positivos y falsos negativos, lo que indica un comportamiento más homogéneo entre ambas clases. La capacidad de Word2Vec para capturar relaciones semánticas entre palabras parece contribuir a esta mejora, reduciendo el sesgo observado en TF-IDF y proporcionando una ligera ventaja en términos de estabilidad del modelo.

#### **4.6.3. RoBERTa**

| Métrica (Test)    | Valor  |
| ----------------- | ------ |
| Accuracy          | 0.7869 |
| Precision (macro) | 0.7624 |
| Recall (macro)    | 0.7653 |
| F1-score (macro)  | 0.7638 |

Como vemos en la tabla en el caso de RoBERTa, la red neuronal alcanza un rendimiento comparable al obtenido con TF-IDF y Word2Vec. Las métricas de clasificación reflejan un comportamiento equilibrado, aunque sin una mejora clara respecto a las otras representaciones.

Las curvas de entrenamiento muestran nuevamente una clara separación entre las métricas de entrenamiento y test, lo que indica que la complejidad de los embeddings contextuales no se traduce directamente en una mejora del rendimiento cuando se utilizan arquitecturas densas estándar. Este resultado sugiere que los embeddings de RoBERTa requieren modelos más especializados o procesos de fine-tuning para explotar plenamente su potencial.

<img width="1120" height="443" alt="image" src="https://github.com/user-attachments/assets/d7a67db4-aea0-4951-bcb5-e3ac8c89d6bb" />

Si nos fijamos en la matriz de confusión recogida en la siguiente tabla:

| Real \ Predicha | 0 (False) | 1 (True) |
|-----------------|-----------|----------|
| 0 (False)       | 148       | 64       |
| 1 (True)        | 69        | 343      |

En el caso de RoBERTa, la matriz de confusión refleja un comportamiento similar al de Word2Vec, con una distribución de errores relativamente equilibrada entre clases. A nivel global, los valores de F1-score son comparables a los obtenidos con otras representaciones, siendo la principal diferencia la mejora en la predicción de la clase minoritaria frente a TF-IDF. Esto sugiere que el mayor contexto semántico capturado por RoBERTa contribuye a una clasificación más consistente.

#### **Comparación final**

Por último, se comparan los tres casos estudiados:

<img width="1287" height="442" alt="image" src="https://github.com/user-attachments/assets/8a005cc7-ed13-4dd9-b3d9-5987763aaf0d" />

Como se resume en la imagen, donde se compara el rendimiento en test de la red neuronal para las distintas vectorizaciones, las diferencias entre TF-IDF, Word2Vec y RoBERTa son relativamente reducidas en términos de accuracy y error rate. En todos los casos, la red neuronal logra resultados competitivos, pero sin superar de forma clara a los modelos lineales analizados previamente.

Estos resultados indican que, para el problema abordado, el incremento de complejidad asociado a una red neuronal profunda no garantiza una mejora significativa del rendimiento. La elección de la representación vectorial sigue siendo un factor clave, y el uso de modelos más complejos debe ir acompañado de estrategias adicionales de regularización o arquitecturas adaptadas al tipo de embedding empleado.

### **4.7. Fine-tuning de RoBERTa**

En este apartado se presentan los resultados obtenidos tras aplicar fine-tuning sobre el modelo RoBERTa-base, adaptándolo directamente a la tarea de clasificación binaria de noticias true/false. A diferencia de los experimentos anteriores, en este caso el modelo se entrena de forma end-to-end, ajustando los pesos del Transformer a partir del conjunto de entrenamiento.

Durante el preprocesado, el texto se tokeniza y se transforma en los campos input_ids y attention_mask, que constituyen la entrada del modelo, junto con las etiquetas correspondientes. El entrenamiento se ha realizado utilizando la librería Hugging Face Transformers, monitorizando la pérdida y las métricas de evaluación a lo largo de las épocas. Una vez hecho esto obtemnemos los siguientes resultados resumidos en la tabla:

| Métrica (Test)    | Valor  |
| ----------------- | ------ |
| Accuracy          | 0.7997 |
| Precision (macro) | 0.79   |
| Recall (macro)    | 0.75   |
| F1-score (macro)  | 0.76   |

El modelo ajustado mediante fine-tuning alcanza una accuracy cercana al 80% en el conjunto de test, situándose entre los mejores resultados obtenidos a lo largo del proyecto. El informe de clasificación muestra un rendimiento especialmente alto en la detección de la clase true, con valores elevados de precision y recall, mientras que la clase false presenta mayores dificultades, algo consistente con el desbalance del dataset.

Los resultados obtenidos confirman que el fine-tuning de RoBERTa permite explotar de forma más efectiva la información contextual capturada por el modelo, superando el rendimiento alcanzado cuando los embeddings de RoBERTa se utilizan únicamente como características de entrada para clasificadores externos. No obstante, la mejora respecto a otros enfoques no es drástica, lo que sugiere que, para este problema concreto, modelos más simples ya capturan gran parte de la información relevante.


## **5.-Proyecto de Extensión**
Como extensión del proyecto principal, y dado que en la introducción del trabajo se planteaba la hipótesis de que las noticias falsas tienden a presentar un lenguaje más negativo o alarmista, se ha llevado a cabo un análisis de sentimiento sobre el contenido textual de las noticias. El objetivo de esta extensión es evaluar si existen diferencias significativas en el sentimiento entre noticias verdaderas y falsas que puedan complementar los resultados de clasificación obtenidos previamente.

Para ello, se han empleado dos enfoques distintos en función del tipo de texto analizado:

- VADER para el análisis de sentimiento en los títulos de las noticias.
- Flair para el análisis de sentimiento en el cuerpo completo de las noticias.

### **5.1. Análisis de sentimiento en títulos con VADER**

En primer lugar, se ha analizado el sentimiento de los títulos de las noticias utilizando la librería VADER, una herramienta basada en léxico y reglas que genera una puntuación continua entre −1 (muy negativo) y 1 (muy positivo). Este análisis resulta adecuado para textos cortos como titulares. Se han obtenido los siguientes resultados:

| Etiqueta | Sentimiento medio |
|----------|-------------------|
| False (0) | -0.0650 |
| True (1)  | -0.0626 |

| Etiqueta | Negativo | Neutral | Positivo |
|----------|----------|---------|----------|
| False (0) | 355 | 480 | 223 |
| True (1)  | 684 | 891 | 486 |

| Etiqueta | Negativo | Neutral | Positivo |
|----------|----------|---------|----------|
| False (0) | 0.3355 | 0.4537 | 0.2108 |
| True (1)  | 0.3319 | 0.4323 | 0.2358 |

El análisis del sentimiento medio por etiqueta (0 = false, 1 = true) muestra valores muy próximos entre ambas clases, con una ligera tendencia negativa en ambos casos. Asimismo, la distribución del sentimiento revela que la mayoría de los títulos se clasifican como neutrales, tanto para noticias verdaderas como falsas, siendo el sentimiento negativo el segundo más frecuente.

Estos resultados indican que, en el caso de los títulos, no se observan diferencias claras de sentimiento entre noticias verdaderas y falsas. La hipótesis inicial de que los títulos de noticias falsas presentan un tono más negativo no puede confirmarse a partir de este análisis.

### **5.2. Análisis de sentimiento en el cuerpo de las noticias con Flair**

Dado que el análisis de sentimiento en títulos no resulta concluyente, se ha extendido el estudio al cuerpo completo de las noticias. Para ello se ha utilizado la librería Flair, más adecuada para textos largos y complejos, ya que emplea modelos de lenguaje entrenados específicamente para clasificación de sentimiento. Se han obtenido los siguientes resultados:

| Etiqueta | Sentimiento medio |
|----------|-------------------|
| False (0) | 0.9613 |
| True (1)  | 0.9580 |

| Etiqueta | Negativo | Positivo |
|----------|----------|----------|
| False (0) | 867 | 191 |
| True (1)  | 1742 | 319 |

| Etiqueta | Negativo | Positivo |
|----------|----------|----------|
| False (0) | 0.8195 | 0.1805 |
| True (1)  | 0.8452 | 0.1548 |

En este caso, el análisis se ha centrado en dos categorías de sentimiento: positivo y negativo, eliminando la clase neutral. Los resultados muestran que, tanto para noticias verdaderas como falsas, entre el 80% y el 85% del contenido presenta un sentimiento negativo, siendo esta la categoría claramente dominante en ambas clases.

Este comportamiento indica que no existe una diferencia significativa en el sentimiento global del texto entre noticias verdaderas y falsas. El lenguaje negativo predomina de forma generalizada, lo cual puede explicarse por el contexto del dataset, centrado en noticias relacionadas con la pandemia de COVID-19, un tema intrínsecamente asociado a situaciones adversas y lenguaje alarmista.

### **Conclusión proyecto de extensión**

Los resultados obtenidos en este proyecto de extensión sugieren que el sentimiento negativo no es un factor discriminante entre noticias verdaderas y falsas en la base de datos analizada. Aunque inicialmente podría esperarse un mayor tono negativo o catastrofista en las noticias falsas, el análisis muestra que este tipo de lenguaje es común a ambas clases.
Por lo tanto, el análisis de sentimiento, al menos en su forma directa, no aporta una separación clara entre clases, lo que refuerza la necesidad de recurrir a enfoques más complejos basados en patrones semánticos y contextuales, como los modelos de clasificación desarrollados en el proyecto principal.

## **6.-Conclusiones**

En este trabajo se ha abordado el problema de la clasificación automática de noticias relacionadas con la COVID-19 como verdaderas o falsas, analizando de forma sistemática el impacto de distintas técnicas de vectorización y modelos de clasificación. Para finalizar, en este apartado vamos a recoger todas las conclusiones que hemos ido mencionando y recopilando a lo largo del proyecto. En la siguiente tabla se muestra una comparativa global de rendimiento usando el accuracy en test:

| Modelo | Vectorización / Enfoque | Accuracy (Test) |
|-------|-------------------------|-----------------|
| Regresión Logística | Word2Vec | 0.8045 |
| SVM lineal | Word2Vec | 0.8029 |
| Red Neuronal | TF-IDF | 0.7949 |
| RoBERTa (Fine-tuning) | End-to-end | 0.7997 |

Los resultados experimentales muestran que es posible alcanzar valores de accuracy cercanos al 80% utilizando enfoques muy diversos, lo que pone de manifiesto la viabilidad del uso de técnicas de Procesamiento del Lenguaje Natural para la detección de desinformación.

Una de las principales conclusiones del estudio es que la representación del texto tiene un impacto tan relevante como el propio modelo de clasificación. Técnicas relativamente simples como TF-IDF y Word2Vec, combinadas con clasificadores lineales como Regresión Logística o SVM, ofrecen un rendimiento muy competitivo, comparable al de modelos más complejos. En particular, Word2Vec proporciona una representación semántica que mejora la estabilidad del modelo y reduce ciertos sesgos observados en enfoques puramente basados en frecuencia.

Por otro lado, los modelos más complejos, como la red neuronal y el fine-tuning de RoBERTa, no garantizan automáticamente una mejora significativa del rendimiento. En el caso de la red neuronal, se observa una mayor capacidad de ajuste acompañada de problemas de sobreajuste, mientras que el fine-tuning de RoBERTa logra explotar mejor la información contextual del texto, aunque con un coste computacional considerablemente mayor. Estos resultados sugieren que, para este problema concreto, el incremento de complejidad debe evaluarse cuidadosamente frente a las ganancias reales obtenidas.

Finalmente, el análisis de sentimiento realizado como proyecto de extensión indica que el lenguaje negativo predomina tanto en noticias verdaderas como falsas, lo que dificulta el uso del sentimiento como un criterio discriminante claro. Este hecho refuerza la idea de que la detección de desinformación requiere modelos capaces de capturar patrones semánticos y contextuales más profundos, más allá de características superficiales como el tono emocional del texto.

### **6.1. Limitaciones**

Este trabajo presenta algunas limitaciones que deben considerarse al interpretar los resultados. En primer lugar, el análisis se ha realizado sobre un dataset específico de noticias relacionadas con la COVID-19, lo que limita la generalización de los modelos a otros dominios informativos. Además, el problema se ha simplificado a una clasificación binaria, eliminando la categoría partially false, lo que reduce la granularidad del análisis en escenarios reales de verificación.

Por otro lado, los modelos más complejos, como la red neuronal y el fine-tuning de RoBERTa, implican un mayor coste computacional y presentan signos de sobreajuste, lo que dificulta su aplicación en entornos con recursos limitados. Asimismo, el análisis de sentimiento realizado no ha mostrado una capacidad discriminante clara entre noticias verdaderas y falsas, limitando su utilidad como característica independiente.

### **6.2. Trabajo futuro**

Como trabajo futuro, sería interesante extender el estudio a una clasificación multiclase, reincorporando la etiqueta partially false, así como evaluar la generalización de los modelos en otros conjuntos de datos y dominios temáticos. Esto permitiría validar la robustez de los enfoques propuestos frente a distintos tipos de desinformación.

Adicionalmente, podrían explorarse arquitecturas neuronales más específicas para embeddings contextuales, técnicas de regularización más avanzadas y métodos de explicabilidad que faciliten la interpretación de las decisiones del modelo. La incorporación de información adicional, como metadatos o fuentes de las noticias, constituye también una línea prometedora para mejorar el rendimiento en aplicaciones reales.














