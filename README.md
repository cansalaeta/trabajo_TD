# **CLASIFICADOR DE NOTICIAS ACERCA DEL COVID-19**

## **1.-Descripción del problema**

Durante la pandemia de COVID-19, la propagación del virus vino acompañada de una enorme cantidad de noticias que se difundían en cuestión de minutos por todo el mundo. Noticias falsas sobre curas milagrosas, efectos secundarios de las candidatas a vacunas que se inventaban o teorías conspirativas acerca de la falsedad de la enfermedad circularon ampliamente, afectando a la percepción pública y mezclando información valiosa con bulos que circulaban por la red.

En este proyecto se han desarrollado diferentes modelos de clasificación automáticos capaces de distinguir entre noticias verdaderas y falsas relacionadas con el Covid-19 mediante técnicas de Procesamiento del Lenguaje Natural (NLP). El reto pasa por conseguir detectar patrones y diferenciar contenido informativo real de contenido engañoso y manipulador.

## **2.-Conjunto de datos**

Para abordar el problema, hemos contado con una base de datos con noticias recopiladas de diferentes fuentes y etiquetadas como 1-True o 0-False/Partially False. Concretamente, hay un total de 3119 noticias, repartidas entre 2061 verdaderas y 1058 falsas, de las que no solo se muestra la noticia y la etiqueta, sino también el título y la categoría (True/False/Partially False). 

En cuanto a esta subdivisión de las noticias falsas entre False y Partially False, dado que el número de noticias verdaderas es casi el doble, hemos podido comprobar que provoca una degradación en las prestaciones de algunos de los algoritmos de clasificación al intentar clasificar entre 3 etiquetas diferentes. Es por ello que se ha simplificado el problema a un clasificador binario de noticias true/false sobre el Covid-19 eliminando la categoría de Partially False.

En cuanto a la longitud de las noticias

<img width="1624" height="573" alt="image" src="https://github.com/user-attachments/assets/e0ed4707-3508-4b86-95c3-74e5e58e23ca" />


Como trabajo previo a la presentación de las técnicas de vectorización y algoritmos clasificadores, se ha realizado una limpieza y homogeneización de este conjunto de noticias para evitar que caracteres, urls y palabras concretas (como cambiar *viru* por *virus*) entren a nuestros modelos e introduzcan ruido que dificulte el trabajo de clasificación.

## **3.-Metodologías utilizadas**

***3.1: Técnicas de vectorización:***

Se ha presentado en el punto anterior el dataset con noticias desde el que partimos para entrenar, validar y testear nuestros modelos. Sin embargo, para entrenar un algoritmo de clasificación como SVM, redes neuronales, etc. necesitamos convertir el texto en representaciones numéricas, para lo cual empleamos 3 técnicas de vectorización diferentes que se presentan a continuación. Destacar que en este apartado del proyecto, tán solo hemos definido y experimentado con estos tres modelos aplicados sobre la base de datos global y será en el siguiente apartado donde dividiremos el dataset en *train*,*test* y *validation*.

- TF-IDF: Esta técnica transforma cada una de nuestras noticias en un vector basándose únicamente en la importancia de las palabras en las noticias. Para implementarlo, se han propuesto 2 alternativas con dos librerías de python: gensim (partiendo de una descripción *Bag of Words*) y un modelo ya definido de sklearn. Las prestaciones en ambos casos hemos comprobado que son prácticamente iguales y por simplicidad, se ha utilizado el modelo de sklearn, que internamente realiza la tokenización de las noticias, vectorización y reducción del vocabulario. Esto es un punto importante, pues el vocabulario original de la base de datos está en torno a 37000 palabras, lo cual es difícil (e innecesario) de manejar por los modelos. Es por ello que se han suprimido aquellas palabras que han aparecido en más del 60% de las noticias o que aparecen en menos de 10.
  
- Word2Vec: Es un modelo que aprende embeddings semánticos de palabras, es decir, vectores densos donde palabras con significados similares tienen representaciones similares. Es una técnica a priori más compleja que TF-IDF (que solo se basa en frecuencia de aparición de las palabras) y esto se ve reflejado en los resultados que se presentarán posteriormente. A diferencia de TF-IDF, que partía de las noticias "limpias y homogeneizadas", en este caso se parte de los tokens de dichas noticias. Igual que antes, se ha realizado una reducción del vocabulario antes de calcular embeddings y para facilitar tareas futuras.

- Embeddings contextuales: Son vectores generados por modelos donde la representación de una palabra depende del contexto en el que aparece, es decir, capturan relaciones complejas entre palabras y son adecuados para representaciones profundas. En nuestro caso, igual que con TF-IDF, se ha realizado una comparativa entre 3 posibles modelos: BERT, RoBERTa y distilBERT. En este caso, para la elección del modelo más adecuado, hemos realizado una pequeña comparativa con una porción de la base de datos para ver cuál es el que ofrece mejores prestaciones. El resultado es que, para nuestra aplicación de clasificador *true*/*false*, es el modelo de RoBERTa el que mejor funciona, luego ha sido el que se ha escogido para los clasificadores que se presentan en el siguiente punto.

***3.2: Modelos de clasificación:***

Las técnicas de vectorización planteadas se han utilizado para adecuar las noticias de nuestra base de datos a entradas válidas que sirvan para 3 modelos que se han evaluado: 2 algoritmos de Scikit-Learn (Regresión Logística y SVM) y una red neuronal que hemos diseñado. Por otra parte, se ha utilizado fine-tunning sobre el modelo de RoBERTa hallado en el anterior punto (aunque aquí no se utilizan vectorizaciones). En cada uno de los casos, se han ido utilizando diferentes métricas de evaluación de resultados destacando *accuracy* para Regresión Logística y SVM y *matriz de confusión*, *f1-score* y *recall* para la red neuronal y modelo con fine-tunning.

Además, se ha dividido la base de datos en 3 conjuntos de *train*,*test* y *validation*, con una cantidad de noticias del 60%, 20% y 20% del total respectivamente. Así, tanto para la red neuronal como para Regresión Logística y SVM, se han utilizado los modelos definidos en el punto anterior aplicados sobre los tres conjuntos, de forma que cada técnica de vectorización tenga su dataset de *train*,*test* y *validation*.

- Regresión Logística: Es un modelo estadístico lineal que predice la probabilidad de que una instancia pertenezca a una clase usando una función sigmoide. A priori es el clasificador más simple de los utilizados, pero como veremos en el siguiente punto, las prestaciones que ofrece son realmente buenas.

- Supporrt Vector Machine (SVM): Es un modelo de aprendizaje supervisado que busca un límite óptimo que separe las clases con el mayor margen posible (en este caso nuestras noticias *true*/*false*). Se ha escogido por ser un modelo bastante diferente del de Regresión Logística y, en principio, más complejos. En el siguiente apartado se ofrecen los resultados obtenidos en comparación con Regresión Logística.
  
- Red Neuronal: En este caso se ha diseñado un modelo compuesto por capas de nodos (neuronas)
  
- RoBERTa+Fine-Tunning: Partiendo del modelo preentrenado de Hugging face de RoBERTa

## **4.-Resultados experimentales**



## **5.-Conclusiones**



