---
date: 2020-05-04T10:42:42-42:00
featured_image: "/images/metrics.jpg"
tags: ["metrics", "machine learning", "classification"]
title: "Métricas essencias para tarefas de classificação"
---


## Introdução

Um problema de classificação em Machine Learning se refere à tarefa de predizer a classe de uma amostra dado suas características ("*features*"). Normalmente, esses problemas são dividos em três categorias: classificação binária, multiclasse e multi-rótulo ("*multilabel*").

A diferença é simples. Quando o problema é formado por apenas duas classes, como por exemplo, predizer se  um livro é de ficção ou não ficção ou se uma notícia é falsa ou não, ele é chamado de classificação binária. De outra forma, quando há três ou mais classes possíveis, o problema pode se tratar de classificação multiclasse ou multi-rótulo.

Tarefas multiclasse acontecem quando só se pode desginar uma classe para cada amostra. Um exemplo disso é o problema clássico de Machine Learning que envolve rotular os dígitos do  [*dataset* MNIST](http://yann.lecun.com/exdb/mnist/). Esse problema envolve predizer um número entre 0 e 9 baseado na imagem de um dígito escrito à mão, ou seja, de 10 classes possíveis, pode-se desingnar apenas um desses números à amostra.

A generalização dessa situação, onde não há limites ao número de classes que pode ser designada a uma amostra, em outras palavras, onde se pode rotular uma amostra com um, três ou cem classes (ou qualquer que seja o número total de classes distintas do problema), é conhecido como classificação multi-rótulo. Ela acontece, por exemplo, quando se quer classificar gêneros musicais de uma banda, o que pode ser feito incluindo mais de uma classe para cada artista. 


## Mas na prática, qual a diferença entre elas?

Basicamente, o que irá diferenciar as categorias de classificação é a forma como o dado em si é representado.

(Os algoritmos usados para realizar predições usando os dados, assim como a forma que suas saídas são produzidas, tendem a variar também, porém isso leva a uma discussão mais complexa. Um bom ponto de partida para se aprofundar nesses detalhes é [essa página](https://scikit-learn.org/stable/modules/multiclass.html) da documentação do sklearn, uma ferramenta bastante usada no universo de ML. Lá, há um resumo excelente sobre estratégias e modelos usados em classificação multiclasse e multi-rótulo.)

Para **problemas binários**, o dado é geralmente representado por "1's" e "0's", sendo "1" o valor para uma amostra do que é conhecido como classe positiva e "0" para a classe negativa, assim:

{{< rawhtml >}}

    <p align="center">
      [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    </p>

{{< /rawhtml >}}

É importante dizer que esses valores podem ser qualquer coisa ("Sim" e "Não", "Bom" e "Ruim", "Azul" e "Vermelho", etc.), contanto que sejam distintos.

Para **problemas multiclasse**, cada rótulo é representado por um valor distinto também, geralmente variando de 0 a n-1, n sendo o número de classes distintas. Uma possível representação do dado para um problema com 5 classes é o seguinte vetor:

{{< rawhtml >}}

    <p align="center">
      [2, 3, 2, 1, 0, 4, 4, 0, 1, 2, 4, 4, 2, 1, 1, 3, 2, 2, 4]
    </p>
{{< /rawhtml >}}

Em **tarefas multi-rótulo**, as coisas se tornam um pouco mais complexas. Nesses problemas, existem duas estratégias geralmente usadas: **Revelância Binária** e **Powerset dos rótulos**.

Na técnica da "Revelância Binária", o dado é formado por um vetor com n posições (n sendo o número de classes distintas). Para cada posição, pode-se atribuir dois valores: um apontando a ocorrência da classe representada por aquele índice naquela amostra e outro indicando ausência. Normalmente, esses valores são "1" e "0". Então, imaginando que existem 5 classes no nosso dado ("0", "1", "2", "3" e "4") e 10 amostras, poderíamos ter:

{{< rawhtml >}}

    <p align="center">
      [[0, 1, 0, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 0], </br>
       [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 1], [1, 1, 1, 0, 1]]
    </p>

{{< /rawhtml >}}

O primeiro item na lista, por exemplo, indica que os rótulos presentes na primeira amostra são "1" e "3".

A opção do **powerset de rótulos** se refere à estratégia de transformar o conjunto de combinações de rótulos presentes no dado representado pelo método da revelância binária em um conjunto de rótulos distintos, isto é, transformar um problema multi-rótulo em um problema multiclasse. Tomando a lista acima como exemplo, teríamos a seguinte transformação:


{{< rawhtml >}}
    <p align="center">
      [0, 1, 0, 1, 0] -> 0 </br>
      [1, 0, 0, 0, 0] -> 1 </br>
      [1, 0, 0, 0, 1] -> 2 </br>
      [1, 0, 1, 0, 1] -> 3 </br>
      [1, 1, 0, 0, 0] -> 4 </br>
      [1, 1, 1, 0, 0] -> 5 </br>
      [1, 1, 1, 0, 1] -> 6 </br>
      [1, 1, 1, 1, 1] -> 7 
    </p>
{{< /rawhtml >}}

E o dado resultante seria:

{{< rawhtml >}}

    <p align="center">
      [0, 4, 7, 2, 5, 1, 5, 4, 3, 6]
    </p>


{{< /rawhtml >}}

Outra importante diferença entre esses três tipos de classificação é a maneira como a qualidade das predições feitas é medida. Cada uma utiliza uma abordam ligeiramente diferente para calcular quão bem o modelo está performando. Vamos dar uma olhada nisso.


### Avaliação da classificação binária

Um das maneiras mais básicas de se medir o quão bem um modelo prediz as classes de novas amostras é utilizando um diagrama conhecido como matriz de confusão. Ele é usado para visualizar como as classes preditas e verdadeiras estão relacionadas. Então, utilizando o exemplo de livros de ficção/não-ficção mencionados previamente, poderíamos ter uma matriz que se parece com isso:


{{< rawhtml >}}
<img src="/images/metrics_classif/conf_matrix_pt.png"  width="80%" style="max-width: 70%;margin: 0% 15%" />
{{< /rawhtml >}}

As linhas agrupam o número de amostras preditas em cada classe. As colunas representam o número real de amostras presentes em cada classe. Mas como realizar conclusões a partir disso?

Primeiramente, é importante se familiarizar com o conceito por trás de verdadeiros/falsos positivos e verdadeiros/falsos negativos.

Verdadeiros positivos se referem a condição onde o modelo corretamente prediz a classe positiva, que em nosso exemplo é "Ficção". De maneira equivalente, um verdadeiro negativo ocorre quando o modelo corretamente classifica uma amostra como sendo da classe negativa ("Não-ficção").

Os dois termos restantes se referem a situações onde o modelo erra. Então, um falso positivo (também conhecido como erro do tipo I) acontece quando a amostra pertence a classe negativa mas é predita como sendo da positiva. Então, quando o rótulo "Ficção" é dado a um livro de não-ficção, um falso positivo ocorre. O oposto, isto é, um falso negativo (ou erro do tipo II), acontece quando uma amostra da classe positiva (um livro que é de ficção) é rotulado incorretamente como um livro de não-ficção.

Com isso em mente, vamos voltar ao nosso exemplo. Temos 100 livros divididos assim:


* 43 livros de ficção;
* 57 livros de não-ficção.

O modelo predisse a maior parte dos livros na classe correta, mas alguns foram classificados de maneira incorreta. Resumindo, existem 42 verdadeiros posivitos, 10 falsos positivos, 1 falso negativo e 47 verdadeiros negativos. 

Ok. Mas como interpretar esse resultado? Ele é bom ou ruim? Precisamos de alguma métricas para realizar uma avaliação.


#### Entendendo acurácia, precisão e revocação (recall)

Em tarefas de classificação, as métricas mais comuns são acurácia, precisão, revocação (usualmente chamada de recall) e f1-score. Elas proveem um valor numérico, de 0 a 1, que avalia a relação entre predições certas e erradas. Aqui as suas fórmulas:


$$ Acurácia = \frac{ TP + TN }{ TP + TN + FP + FN }$$
$$ Precisão = \frac{ TP }{ TP + FP }$$
$$ Recall = \frac{ TP }{ TP + FN }$$
$$ F\_{1} = \frac{ 2 \cdot Precisão \cdot Recall }{ Precisão + Recall } $$

Cada um usa uma abordagem diferente para analisar os resultados obtidos, ponderando as predições corretas e errôneas de maneiras levemente diferentes.

A **acurácia** representa quantas predições do total foram corretas:

$$ Accurácia = \frac{ 42 + 47 }{42 + 47 + 10 + 1} = \frac{89}{100} = 89 \\% $$


A **precisão** mede a porcentagem de classificações corretas de amostras posivitas em relação ao total de amostras classificadas como sendo da classe positiva. 


$$ Precisão = \frac{ 42 }{42 + 10} = \frac{42}{52} \approx 81 \\% $$

O **recall** segue uma abordagem semelhante à precisão, mas em vez de usar o total de amostras rotuladas como sendo da classe positiva para calcular a porcentagem, ele retorna a porcentagem de classificações corretas de amostras positivas em relação ao número total de amostras realmente positivas. Aqui está seu valor: 

$$ Recall = \frac{ 42 }{42 + 1} = \frac{42}{43} \approx 98 \\% $$

A última forma apresentada, conhecida como **F1 score**, avalia a [média harmônica](https://pt.wikipedia.org/wiki/M%C3%A9dia_harm%C3%B4nica) da precisão e do recall. Para este exemplo, seu valor é:



$$ F\_{1} = \frac{ 2 \cdot 0.81 \cdot 0.98 }{0.81 + 0.98} = \frac{1.58}{1.79} \approx 88 \\% $$

Ok. Agora temos 4 valores e todos os resultados parecem estar ótimos, certo?

Todos estão perto de 100%. A acurácia, por exemplo, nos diz que de 100 classes preditas, quase 90% foram corretas. 

Aparentemente, independentemente do problema com quem alguém estivesse lidando, esses números parecem ser o bastante para dizer que o nosso modelo está perto do melhor possível. Isso é realmente verdade? Vamos ver.

Imagine que em vez de classificar livros, o modelo estivesse predizendo se o tipo de câncer de um paciente é benigno ou maligno. Mantendo os número iguais, de 43 pessoas que teriam câncer, 1 seria classificada de forma errada. Isso não é um problema tão grave. O teste poderia ser repetido para verificar o resultado. O grande problema é com os falsos positivos. De 57 pessoas com uma forma de câncer maligno, 10 (20% aproximadamente) receberiam um resultado dizendo que elas possuem uma forma de câncer menos agressiva. Isso é ruim. Imagine quantas dessas pessoas deixariam de repetir o exame e teriam consequências graves por isso.

> Um número "cru" em si não é suficiente para definir o quão bom uma predição é. O contexto do problema tem uma enorme importância em determinar as métricas a serem usadas e o que é considerado "bom" ou "ruim".

Às vezes, mais falsos negativos que falsos positivos, como no exemplo do câncer benigno/maligno, e consequentemente um recall menor que a precisão, é preferível do que o contrário. Em outras situções, atingir uma acurácia de 70% (que é melhor do que um chute com 50% de acerto) será excelente. Cada problema é único e o papel de um entusiasta ou profissional de Machine Learning é entender as métricas e usar a mais apropriada em cada contexto.

### Avaliação da classificação multiclasse

Imagine que se decida que, em vez de classificar os livros em ficção ou não-ficção, haverá uma classificação baseada no número de estrelas que o livro recebeu em uma loja online. Então, supondo que existam 5 classes (1 a 5 estrelas), uma matriz de confusão possível para esse problema seria assim:

{{< rawhtml >}}
<img src="/images/metrics_classif/conf_matrix2_pt.png"  width="50%" style="max-width: 50%;margin: 0% 25%" />
{{< /rawhtml >}}

Repare que ela continua possuindo os 100 livros, mas existem 5 colunas e 5 linhas. Além disso, os livros estão classificados de acordo com o seu número de estrelas agora. Como os verdadeiros/falsos positivos serão medidos agora sem a existência das classes "positiva" e "negativa"? Vamos dar uma olhada.

A acurácia é a métrica mais fácil de ser calculada. Como dito anteriormente, ela representa a divisão do número total de predições corretas pelo número total de amostras. Sendo assim, ao se olhar a matriz acima, todos os números na diagonal com o azul mais escuro correspondem as predições corretas porque eles representam as situações onde o modelo atribuiu a classe correta àquela amostra. Para este problema, a acurácia é:


$$ Accurácia = \frac{ 15 + 14 + 10 + 12 + 17 }{100} = \frac{68}{100} = 68 \\% $$

Ok. Mas como calcular os valores de precisão e recall? Eles dependem do número de falsos positivos/negativos e consequentemente da existência das classes positiva/negativa.

Um problema multiclasse pode ser tratado com uma extensão da classificação binária. Nós apenas precisamos considerar o problema como uma ["coleção de problemas binários, um para cada classe"](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel), isto é, identificar se um livro tem 1 a 5 estrelas é o mesmo que descobrir se ele deveria ter 1 estrela ou não. Se sim, ele obviamente tem 1 estrela e o problemas está resolvido. Caso contrário, ele terá 2 estrelas ou pertence a alguma das classes subsequentes (3, 4 ou 5). Se não tiver 2, terá 3 estrelas ou mais. E assim por diante. Dessa forma, dividimos a tarefa de classificar uma amostra em uma de 5 classes em 5 problemas binários e somos capazes de calcular as métricas para cada classe. Então, só necessitamos combinar ou calcular a média das métricas inviduais em uma só, produzindo o resultado final.

Assim, vamos considerar as amostras com 1 estrela, por exemplo. Trataremos as outras classes como "2 ou mais". A matriz de confusão ficaria assim:

{{< rawhtml >}}
<img src="/images/metrics_classif/2_more_pt.png"  width="70%" style="max-width: 70%;margin: 0% 15%" />
{{< /rawhtml >}}

Agora, calculando precisão, recall e f1-score para a classe "1 estrela", nós temos:

$$ Precisão = \frac{ 15 }{ 15 + 3} = \frac{15}{18} \approx 83 \\% $$
$$ Recall = \frac{ 15 }{15 + 5} = \frac{15}{20} = 75 \\% $$
$$ F\_{1} = \frac{ 2 \cdot 0.83 \cdot 0.75 }{ 0.83 + 0.75}  \approx 79 \\% $$

Repetindo esse processo com cada uma das classes restantes, obtém-se um total de 5 valores para cada uma dessas métricas, uma para cada estrela, como mostrado abaixo:

$$ Precisão\_{1estrela} = \frac{ 15 }{ 18 }  \approx 83 \\% \quad\quad 
Recall\_{1estrela} = \frac{ 15 }{ 20 }  = 75 \\% \quad\quad
F\_{11estrela} = \frac{ 2 \cdot 0.83 \cdot 0.75 }{ 0.83 + 0.75}  \approx 79 \\% $$

$$ Precisão\_{2estrelas} = \frac{ 14 }{ 24 }  \approx 58 \\% \quad\quad 
Recall\_{2estrelas} = \frac{ 14 }{ 16 }  \approx 87 \\% \quad\quad
F\_{12estrelas} = \frac{ 2 \cdot 0.58 \cdot 0.87 }{ 0.58 + 0.87}  \approx 70 \\% $$

$$ Precisão\_{3estrelas} = \frac{ 10 }{ 14 }  \approx 71 \\% \quad\quad 
Recall\_{3estrelas} = \frac{ 10 }{ 24 }  \approx 42 \\% \quad\quad
F\_{13estrelas} = \frac{ 2 \cdot 0.71 \cdot 0.42 }{ 0.71 + 0.42}  \approx 53 \\% $$


$$ Precisão\_{4estrelas} = \frac{ 12 }{ 22 }  \approx 54 \\% \quad\quad 
Recall\_{4estrelas} = \frac{ 12 }{ 12 }  = 100 \\% \quad\quad
F\_{14estrelas} = \frac{ 2 \cdot 0.54 \cdot 1 }{ 0.54 + 1}  \approx 70 \\% $$


$$ Precisão\_{5estrelas} = \frac{ 17 }{ 22 }  \approx 77 \\% \quad\quad 
Recall\_{5estrelas} = \frac{ 17 }{ 28 }  \approx 60 \\% \quad\quad
F\_{15estrelas} = \frac{ 2 \cdot 0.77 \cdot 0.6 }{ 0.77 + 0.6}  \approx 67 \\% $$

-------------------------------------------------------------------------------

## Calculando precisão e recall usando interseção de conjuntos

É interessante mostrar que precisão e recall também podem ser calculados utilizando interseção de conjuntos. Considere alguns vetores representando classes verdadeiras e preditas como dois conjuntos com os rótulos:


{{< rawhtml >}}
    <p align="center">
      Verdadeiro -> [1, 2, 0, 3, 4, 1, 1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 0, 1, 2] </br>
      Predito &nbsp;&nbsp;&nbsp;&nbsp; -> [1, 0, 0, 3, 1, 1, 0, 1, 1, 1, 1, 3, 0, 2, 1, 4, 0, 0, 2]
    </p>

{{< /rawhtml >}}

Precisão e recall podem ser calculados para cada classe assim:

$$ Precisão = \frac{|A \cap B|}{|B|} $$
$$ Recall = \frac{|A \cap B|}{|A|} $$

**A** armazena todos os valores de uma classe específica ocorrendo no vetor de classes verdadeiras.

**B** armazena todos os valores de uma classe específica ocorrendo no vetor de classes preditas.

Então, para a classe "1", por exemplo:


{{< rawhtml >}}

    <p align="center">
      Verdadeiro ->
        [<span style="color: green;font-weight:bold">1</span>, 
        2, 
        0, 
        3, 
        4, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        0, 
        0, 
        3, 
        3,
        2, 
        2, 
        4, 
        4, 
        0, 
        <span style="color: green;font-weight:bold">1</span>, 
        2] </br>
      Predito &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ->
        [<span style="color: red;font-weight:bold">1</span>, 
        0, 
        0, 
        3, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        0, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        3,
        0, 
        2, 
        <span style="color: red;font-weight:bold">1</span>, 
        4, 
        0, 
        3, 
        2] </br>
    </p>
    <p align="center">
      A -> [1, 1, 1, 1, 1] </br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B -> [1, 1, 1, 1, 1, 1, 1, 1]
    </p>

{{< /rawhtml >}}

Então, a interseção de **A** e **B** representa o número de predições da classe "1" que são corretas, isto é, verdadeiros positivos.

O número de elementos em **A** representa o número de amostras da classe positiva, nesse caso, a classe "1".

O número de elementos em **B** representa o número de predições nas quais foi atribuído o rótulo da classe positiva ("1", nesse caso).

Com isso em mente, é claro ver porque as fórmulas acima nos permite calcular precisão e recall. Estamos dividindo o número de verdadeiros positivos pelo número de verdadeiros positivos mais o número de falsos positivos (amostras que tiveram "1" atribuídas a si mas são de outra classe), ou seja, a precisão; e dividindo os verdadeiros positivos pelo número de amostras que são na verdade da classe "1", ou seja, o recall.

-----------------------------

Agora, existem 15 valores representando a performance das predições. É hora de combiná-los para produzir um único valor para cada uma das métricas. Estas estratégias usam o cálculo envolvendo conjuntos mostrado acima.

## Calculando as médias da precisão, do recall e do f1-score

Como mostrado acima, em tarefas de classificação com mais de duas classes, cada métrica vai ter um valor por classe. Mas problemas do mundo real podem exceder as dezenas de classe facilmente e entender a performance de um modelo com centenas de valores se torna incrementalmente difícil. Por conta disso, algumas estratégias para combinar os resultados de cada classe existem. Problemas multiclasse normalmente usam três maneiras para realizar essa combinação. Elas são muito comuns (o scikit-learn as suportam, por exemplo) e simples de serem entendidas. Vamos dar uma olhada nelas.


### Micro average

Das 3 estratégias, a única que não usa um somatório é a *micro average*. Seu cálculo basicamente usa o conceito de interseção de conjuntos apresentado previamente para determinar o valor de cada métrica. De acordo com a documentação do scikit-learn, "utilizar "micro-averaging" pode ser preferível em configurações multi-rótulo, incluindo classificação multiclasse onde uma classe majoritária deve ser ignorada".

Sua definição basicamente é:

$$ Micro \\: average = M(y, \hat(y)) $$

Onde:
* *M* é a métrica;
* *y* é o conjunto de rótulos verdadeiros;
* *ŷ* is the set of predicted labels.

Em nosso exemplo, os vetores com os rótulos verdadeiros e preditos se parece com:

{{< rawhtml >}}

    <p align="center">
      Actual &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> 
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] </br>
      Predicted &nbsp;-> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 3, 3, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      3, 3, 3, 3, 3, 3, 1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, </br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    </p>
{{< /rawhtml >}}

A interseção (Verdadeiro ∩ Predito) é representada pelos elementos iguais na mesma posição em ambos os vetores. O resultado é:

{{< rawhtml >}}

    <p align="center">
      Actual ∩ Predicted -> 
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]</br>
    </p>

{{< /rawhtml >}}

Então, as métricas usando micro-average são:

$$ Precisão\_{Micro} = \frac{ |Verdadeiro \cap Predito|}{ |Predito| } = \frac{68}{100} = 68\\% $$ 
$$ Recall\_{Micro} = \frac{ |Verdadeiro \cap Predito|}{ |Verdadeiro| } = \frac{68}{100} = 68\\% $$ 
$$ F\_{1Micro} = \frac{ 2 \cdot 0.68 \cdot 0.68 }{ 0.68 + 0.68} = \frac{0.924}{1.36} \approx 68 \\% $$


 

### Macro average

A segunda maneira de combinar as métricas é usando "macro-average" nelas. Essa técnica funciona simplesmente calculando a média dos valores dando o mesmo peso para todas as classes. De acordo com a [documentação do sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html), em problemas onde há classes com uma frequência baixa, o uso de "macro-averaging" pode ressaltar a perfomance delas. Entretanto, dar o mesmo nível de importância para todas as classes é frequentemente errado. Dessa forma, o resultado final pode ser tal que a perfomance baixa de classes não frequentes pode ser enfatizado demais.

Embora existam problemas ao se usar macro-average em contextos onde o dado é distribuído de maneira não uniforme, essa estratégia pode ser usada em casos onde não há uma disparidade tão grande entre o número de amostras de cada classe.

Abaixo, a sua fórmula:

$$
Macro \\: average = \frac{1}{|L|} \sum\_{l \in L} M(y\_{l}, \hat{y\_{l}})
$$

Onde:

* *L* é o conjunto de rótulos;
* *M* é a métrica;
* *y* é o conjunto de rótulos verdadeiros;
* *ŷ* é o conjunto de rótudos preditos;
* {{< rawhtml >}} <i>y<sub>l</sub></i> {{< /rawhtml >}} é o subconjunto de  *y* com rótulo *l*;
* {{< rawhtml >}} <i>ŷ<sub>l</sub></i> {{< /rawhtml >}} é o subconjunto de *ŷ* com rótulo *l*.

Para o nosso exemplo, utilizar macro-average nas métricas provê os seguintes resultados:

$$ Precisão\_{Macro} = \frac{ 0.83 + 0.58 + 0.71 + 0.54 + 0.77 }{ 5 } \approx 69\\% $$ 
$$ Recall\_{Macro} = \frac{ 0.75 + 0.87 + 0.42 + 1 + 0.6}{ 5 } \approx 73\\% $$ 
$$ F\_{1Macro} = \frac{ 0.79 + 0.7 + 0.53 + 0.7 + 0.67 }{ 5 }  \approx 68 \\% $$


### Média ponderada (Weighted average)

A média ponderada combina as métricas de uma maneira que a frequência de cada classe nas amostras verdadeiras é levada em conta, ou seja, o resultado da métrica para cada classe é ponderado de acordo com sua presença no vetor de rótulos verdadeiros. Usando esta técnica, classes que têm mais amostras tem uma influência maior no resultado final do que classes com menos amostras. Sendo assim, o resultado é calculado proporcionalmente a distribuição das classes.

A fórmula para essa estratégia é:

$$
Weighted \\: average = \frac{1}{\sum\_{l \in L} |y\_{l}|} \sum\_{l \in L} |y\_{l}| M(y\_{l}, \hat{y\_{l}})
$$

Onde:

* *L* é o conjunto de rótulos;
* *M* é a métrica;
* *y* é o conjunto de rótulos verdadeiros;
* *ŷ* é o conjunto de rótudos preditos;
* {{< rawhtml >}} <i>y<sub>l</sub></i> {{< /rawhtml >}} é o subconjunto de  *y* com rótulo *l*;

Para nosso exemplo, combinar as métricas usando a média ponderada resulta em:

$$ Precisão\_{Weighted} = \frac{1}{ 100 } \cdot (20 \cdot 0.83 + 16 \cdot 0.58 + 24 \cdot 0.71 + 12 \cdot 0.54 + 28 \cdot 0.77) \approx 71\\% $$ 
$$ Recall\_{Weighted} = \frac{1}{ 100 } \cdot (20 \cdot 0.75 + 16 \cdot 0.87 + 24 \cdot 0.42 + 12 \cdot 1 + 28 \cdot 0.6) \approx 68\\% $$ 
$$ F\_{1Weighted} = \frac{1}{ 100 } \cdot (20 \cdot 0.79 + 16 \cdot 0.7 + 24 \cdot 0.53 + 12 \cdot 0.7 + 28 \cdot 0.67) \approx 68\\% $$ 


## Avaliação da classificação multi-rótulo

O último tipo de tarefa de classificação é chamada de multi-rótulo (multilabel), onde cada amostra pode ser associada com mais de um rótulo. Um exemplo de classificação multi-rótulo são as *tags* de gêneros musicais em plataformas como a [last.fm](https://www.last.fm).

Como mostrado anteriormente, existem duas maneiras comuns de realizar classificação multi-rótulo: revelância binária e powerset de rótulos. Vamos focar em avaliar o problema usando a primeira abordagem porque é a forma mais comum de lidar com tarefas multi-rótulo e provê uma maneira eficiente e flexível de treinar e testar o modelo.

Então, imagine que se deseje criar um modelo para atribuir *tags* de gênero para algumas bandas automaticamente. Como existem incontáveis gêneros musicais, para simplificar, vamos usar apenas cinco deles (Eletrônica, Folk, Instrumental, Rap e Rock). Baseado nas *tags* providas pela last.fm, o dado seria assim:

{{< rawhtml >}}
<style>
table, th, td {
	width: 100%;
  border: 1px solid #bebebe;
  border-collapse: collapse;
}
th, td {
  padding: 15px;
}
</style>

{{< /rawhtml >}}

| Banda/Artista        | Eletrônica | Folk  | Instrumental | Rap | Rock |
|--------------------|------------|-------|--------------|-----|:----:|
| Bon Iver           |     1      |   1   |      0       | 0   |    0 |
| Beastie Boys       |     0      |   0   |      1       | 1   |    1 |
| Linkin Park        |     1      |   0   |      1       | 1   |    1 |
| twenty one pilots  |    1       |   0   |    0         |  1  |  1   |
| ...                |    ...     |  ...  |    ...       | ... |  ... |

Dessa forma, a saída do modelo para cada amostra seria composta de cinco valores, um para cada gênero que está se predizendo. Mas como avaliar esse resultado? Vamos ver alguns métodos.

### Avaliando resultados de modelos multi-rótulo

#### Samples average

Esse método é uma outra forma de calcular a média das três métricas mostradas anteriormente (precisão, recall e f1-score), mas é destinada para tarefas multi-rótulo apenas. Ele funciona calculando a métrica para cada par de valores verdadeiros e preditos para cada amostra no dado de teste, resultando em um valor único. A fórmula é:

$$
Samples\\: average = \frac{1}{|S|} \sum\_{s \in S} M(y\_{s}, \hat{y\_{s}})
$$

Onde:

* *S* é o conjunto de amostras;
* *M* é a métrica;
* {{< rawhtml >}} <i>y<sub>s</sub></i> {{< /rawhtml >}} é a s-ésima amostra de *y*;
* {{< rawhtml >}} <i>ŷ<sub>s</sub></i> {{< /rawhtml >}} é a s-ésima amostra de *ŷ*.

Então, para as quatro bandas acima, um modelo poderia ter como saída algo como:

| Banda/Artista      | Eletrônica  | Folk  | Instrumental | Rap | Rock |
|--------------------|------------|-------|--------------|-----|------|
| Bon Iver           |     1      |   0   |      0       | 1   |    0 |
| Beastie Boys       |     0      |  1    |      1       | 0   |    1 |
| Linkin Park        |     1      |   0   |      1       | 1   |    1 |
| twenty one pilots  |    1       |   1   |    0         |  1  |  1   |

E para cada amostra, precisão, recall e f1-score são calculados. Então, para a primeira amostra ("Bon Iver"), as métricas são:

$$ Precisão\_{Bon\\:Iver} = \frac{TP}{TP + FP} = \frac{|\\{1,1,0,0,0\\} \cap \\{1,0,0,1,0\\}|}{|\text{n para cada n} \in \\{ 1, 0, 0, 1, 0 \\} \text{onde n = 1}|} =\frac{1}{2} = 50 \\% $$
$$ Recall\_{Bon\\:Iver} = \frac{TP}{TP + FN} = \frac{|\\{1,1,0,0,0\\} \cap \\{1,0,0,1,0\\}|}{|\text{n para cada n} \in \\{ 1, 1, 0, 0, 0 \\} \text{onde n = 1}|} =\frac{1}{2} = 50 \\% $$
$$ F\_{1Bon\\:Iver} = \frac{2 \cdot 0.5 \cdot 0.5}{0.5 + 0.5} = 50 \\% $$

Calculando as métricas para os outros três artistas e combinando-as:

$$ Precisão\_{Samples} = \frac{1}{4} (0.5 + 0.667 + 1 + 0.75) \approx 73 \\% $$
$$ Recall\_{Samples} = \frac{1}{4} (0.5 + 0.667 + 1 + 1) \approx 79 \\% $$
$$ F\_{1Samples} = \frac{2 \cdot 0.73 \cdot 0.79 }{0.73 + 0.79} \approx 76 \\% $$

#### Hamming Loss

Hamming Loss é outra estratégia para availar o resultado de modelos multi-rótulo. Ela é definida como a fração de rótulos que são erroneamente preditos. Então, ela basicamente checa quantos valores no vetor de predições são diferentes dos valores reais.

A fórmula é:

$$
Hamming\\:Loss = \frac{1}{n\_{labels}} \sum\_{j = 0}^{n\_{labels} - 1} 1, \text{if} \\: y\_{j} \neq \hat{y\_{j}} 
$$

Para o nosso exemplo, os vetores verdadeiro e predito são assim:


{{< rawhtml >}}
    <p align="center">
      Actual &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;->
        [<span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>
        ] </br>
      Predicted ->
        [<span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: red;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">0</span>, 
        <span style="color: green;font-weight:bold">1</span>, 
        <span style="color: green;font-weight:bold">1</span>
        ] </br>
    </p>

{{< /rawhtml >}}
E o valor de Hamming Loss:

$$ Hamming \\: Loss = \frac{5}{20} = 0.25 = 25 \\% $$

# Conclusão

Esse artigo deu uma introdução a alguns dos conceitos essencias envolvendo a avalição de problemas de classificação.
Infelizmente, nem toda métrica existente foi mostrada aqui. Existem dezenas delas que nos ajudam a entender e analisar
a performance de um algoritmo de Machine Learning. Mas entender profundamente a diferença entre os três tipos de classificação,
o significado de Verdadeiros/Falsos Positivos/Negativos e como eles se relacionam já é um grande passo para ser capaz de compreender
tópicos mais avançados.

Espero que as três principais métricas usandas em classificação (precisão, recall e f1-score) estejam mais claras após ler esse artigo.
E como sempre, qualquer tipo de feedback é bem-vindo.

# Referências e leituras adicionais 

* [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) (Documentaçãp)
* [Classification: True vs. False and Positive vs. Negative](https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
) (Google ML Crash Course)
* [Machine Learning Fundamentals: The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) (Vídeo no YouTube)
* [Deep dive into multi-label classification..!](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff) (Artigo do Medium)
