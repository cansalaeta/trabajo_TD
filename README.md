# CLASIFICADOR DE NOTICIAS ACERCA DEL COVID-19

**Descripción del problema**

Durante la pandemia de COVID-19, la propagación del virus vino acompañada de una enorme cantidad de noticias que se difundían en cuestión de minutos por todo el mundo. Noticias falsas sobre curas milagrosas, efectos secundarios de las candidatas a vacunas que se inventaban o teorías conspirativas acerca de la falsedad de la enfermedad circularon ampliamente, afectando a la percepción pública y mezclando información valiosa con bulos que circulaban por la red.

En este proyecto se han desarrollado diferentes modelos de clasificación automáticos capaces de distinguir entre noticias verdaderas y falsas relacionadas con el Covid-19 mediante técnicas de Procesamiento del Lenguaje Natural (NLP). El reto pasa por conseguir detectar patrones y diferenciar contenido informativo real de contenido engañoso y manipulador.

**Conjunto de datos**

Para abordar el problema, hemos contado con una base de datos con noticias recopiladas de diferentes fuentes y etiquetadas como 1-True o 0-False/Partially False. Concretamente, hay un total de 3119 noticias, repartidas entre 2061 verdaderas y 1058 falsas, de las que no solo se muestra la noticia y la etiqueta, sino también el título y la categoría (True/False/Partially False). 

En cuanto a esta subdivisión de las noticias falsas entre False y Partially False, dado que el número de noticias verdaderas es casi el doble que el de las falsas, hemos podido comprobar que provoca una degradación en las prestaciones de algunos de los algoritmos de clasificación al intentar clasificar entre 3 etiquetas diferentes. Es por ello que se ha simplificado el problema a un clasificador binario de noticias true/false sobre el Covid-19.

Como trabajo previo a la presentación de las técnicas de vectorización y algoritmos clasificadores, se ha realizado una limpieza y homogeneización de este conjunto de noticias para evitar que caracteres y palabras concretas entren a nuestros modelos e introduzcan ruido que dificulte el trabajo de clasificación.










