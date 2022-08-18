# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:05:49 2022

@author: Anderson Almeida
"""

import streamlit as st
from PIL import Image


diretorio = r'S:\Área de Trabalho'

st.title('Metodologia')

url = 'https://discuss.streamlit.io/t/how-to-create-a-dynamic-clicable-hyperlink-or-button-in-streamlit/12959'

st.write('''
         
Este trabalho utilizada dados fotométricos do projeto Gaia EDR3(ESA), 
que possibilitou a criação de um catálogo de parâmetros fundamentais para aglomerados abertos com 
maior precisão. Este feito nos permitiu, neste momento, estudar a massa dos aglomerados abertos com
mais detalhes e para isso desenvolvemos um novo método de determinação de massas individuais usando 
aglomerados gerados sinteticamente a partir de isócronas teóricas.

''')

st.title('Catálogo de aglomerados abertos')

st.write('''
O catálogo GAIA EDR3 (Gaia Collaboration et al., 2021) é a terceira publicação de
dados da missão e contém astrometria e fotometria com 𝐺 ≤ 21 para aproximadamente
1.8 bilhões de fontes luminosas. Como consequência da alta qualidade de dados, estas
informações possibilitam a determinação e caracterização de membros de milhares de
aglomerados abertos (Cantat-Gaudin et al., 2018); (Soubiran et al., 2018); (Monteiro et
al., 2021), levando a CMDs bem definidos e, portanto, permitindo a determinação dos
parâmetros fundamentais com maior precisão.

Dessa maneira, utilizando o catálogo EDR3 do Gaia, um recente catálogo de aglo-
merados abertos foi elaborado no trabalho de (Dias et al., 2021), selecionando 1743 objetos
com seus membros determinados a partir dos estudos de (Cantat-Gaudin; Anders, 2020),
(Castro-Ginard et al., 2020), (Liu; Pang, 2019), (Sim et al., 2019), (Monteiro et al., 2020)
e (Ferreira et al., 2020). Um dos principais objetivos do trabalho era a determinação
dos parâmetros fundamentais desses aglomerados. Para isso, foram realizados ajustes de
isócronas, utilizando um método de otimização desenvolvido em (Monteiro et al., 2017).


Da amostra inicial contendo 1743 aglomerados, foi selecionada uma subamostra através
de análise visual, em que foram retirados DCMs que exibiam I) poucas estrelas dispostas
no diagrama, II) muitas estrelas distantes do ajuste da isócrona e III) isócronas mal
ajustadas. Essa análise resultou em uma amostra final de 900 aglomerados abertos.

Esta subamostra de aglomerados pode ser consultados na página "Resultados"
jutamente de seus parâmetros fundamentais e outros determinados por nós.

''')


image = Image.open(diretorio +'\Dashboard_Ocs\images\distribuicao_aglomerados.png')

st.image(image, caption='Distribuição galáctica dos 900 aglomerados abertos selecionados neste trabalho. O gráfico apresenta os aglomerados em coordenadas galácticas. A cor é proporcional à idade, sendo azul um aglomerado jovem, verde de idade intermediária e amarelo um aglomerado velho')


st.title('Aglomerados Sintéticos')

st.write('''

A partir dos parâmetros fundamentais dos aglomerados abertos, torna-se viável a
determinação de massa das estrelas membro e, portanto, da massa do aglomerado. 
Para isso, propomos um novo método de determinação de massas, no entanto para descrever o novo método, precisamos descrever
a sua ferramenta fundamental: aglomerados sintéticos. Chamamos aglomerados sintéticos 
os aglomerados criados por um script em python desenvolvido pelo Grupo de Aglomerados Abertos da UNIFEI.

Neste programa temos a autonomia de definir diversos parâmetros como idade, distância, 
metalicidade, avermelhamento, fração de binárias e número de estrelas para compor
o aglomerado. Além disso, temos informações sobre a massa individual de cada estrela
membro e suas magnitudes nos filtros 𝐺, 𝐺𝐵𝑃 e 𝐺𝑅𝑃 do Gaia EDR3.

O processo de geração do aglomerado sintético segue os seguintes passos:
    ''')
    
st.write('🔹 Definição dos parâmetros fundamentais idade e metalicidade;')
st.write('🔹 Busca em um grid a respectiva isócrona teórica para estes parâmetros;')
st.write('🔹Produção de amostra de um número predefinido de estrelas nesta isócrona usando a FMI de Chabrier como distribuição de probabilidades;')
st.write('🔹 Adição de binárias conforme a fração predefinida;')
st.write('🔹 Simulação predefinida de distância e de avermelhamento das estrelas;')
st.write('🔹 Adição de erros fotométricos segundo a definição dos dados Gaia EDR3;')
st.write('🔹 Distribuição espacial das estrelas de acordo com um perfil de King.')

image2 = Image.open(diretorio +'\Dashboard_Ocs\images\Gráfico-sintéticos.png')
st.image(image2, caption='Quatro aglomerados sintéticos plotados em coordenadas RA e DEC para duas idades, distâncias e avermelhamentos.')


st.write('''
Para reproduzir um aglomerado observável, introduzimos os erros fotométricos do
Gaia e uma lei de extinção segundo (Fitzpatrick et al., 2019). Além disso, inserimos um
limite de magnitude aparente no filtro 𝐺 de 19, pois acima desse valor os erros fotométricos
do Gaia são relativamente altos e podem induzir incertezas.
Com todos esses atributos discutidos, montamos um grid de aglomerados sintéticos
que abrange os seguintes parâmetros:
    
''')


st.write('🔹 Idade: de log(age) = 6.6 até log(age) = 10.13 dex;')
st.write('🔹 Distância: de 0.0 até 5 kpc;')
st.write('🔹 Avermelhamento: de 0.0 até 5.0 mag;')
st.write('🔹 Limite de massa: 0.09 𝑀⊙ para a massa inferior e sem limite para massa superior;')
st.write('🔹 [Fe/𝐻]: de -0.90 até +0.70 dex − os sintéticos apresentados neste capítulo e para futuros testes possuem metalicidade solar.')

         
st.title('Um novo método de determinação de massas individuais')

st.write('''
Considerando o que foi discutido, propomos um novo método de determinação
de massa das estrelas membro dos aglomerados abertos, tendo como base os aglomera-
dos sintéticos. Referenciamo-nos a esse método como Determinação de Massa Individual
(doravante DMI). Fundamentalmente, o processo efetua uma comparação dos dados fo-
tométricos observados com os gerados para o aglomerado sintético. Em relação a esse
último, conhecemos a massa equivalente para cada estrela. As etapas do novo método
serão explicadas a partir de um exemplo em que o aglomerado sintético observado possui
as seguintes características: log(age) = 8.5, distância = 1.0 kpc, avermelhamento = 1.0
mag, metalicidade solar e com 300 estrelas observáveis. Essas etapas serão detalhadas a
seguir:
    ''')
    
st.write('''
🔹 I) A partir da geração do aglomerado sintético observado com os parâmetros citados,
é sobreposto um aglomerado sintético de 10000 estrelas com os mesmos parâmetros
fundamentais predeterminados (idade, distância, avermelhamento e metalicidade)
do aglomerado observado para o qual queremos determinar a massa. Na figura abaixo,
estas estrelas estão representadas por um círculo e sua escala de cor varia conforme
sua massa, sendo as menos massivas em vermelho, as intermediárias em verde e as
mais massivas em azul. Perceba pela região do DCM, onde foi realizado o zoom, que
as estrelas observadas (em cinza) se sobrepõem a uma gama de estrelas sintéticas.
''')


image2 = Image.open(diretorio +'\Dashboard_Ocs\images\exemplo_metodo01.png')
st.image(image2, caption='Exemplo de determinação de massa com um aglomerado sintético. Geramos um aglomerado sintético de log(age) = 8.5, distância = 1.0 kpc, avermelhamento = 1.0 magnitude, de metalicidade solar e com 300 estrelas observáveis − estas últimas são representadas pelas marcações "+"em cinza.')

st.write('''
     A partir das massas individuais das estrelas membro dos aglomerados observados,
podemos obter sua massa total. Para realizar isso, construímos um histograma para a dis-
tribuição de massas com a função numpy.histogram da biblioteca numpy. Nele, o número
de intervalos é definido de forma automática pelo algorítimo da biblioteca, que decide o
melhor valor baseado na Regra de Sturges (STURGES, 1926).
Para realizar a integração, ajustamos uma função segmentada composta de duas
retas que nos possibilita encontrar as inclinações de alta (𝛼𝐴) e baixa (𝛼𝐵 ) massa, junta-
mente do 𝑀𝑐, onde acontece o pico de massa − o qual é determinado pela interseção das
retas, como observado na figura abaixo.




\\ imagem aqui


Na figura 24, temos um exemplo do ajuste das retas à distribuição de massa para
o aglomerado sintético de idade intermediária com 300 estrelas. As barras de erro são
relativas à contagem de estrelas em cada intervalo. Acima da figura temos os resultados
das inclinações juntamente do pico de massa.
A massa total do aglomerado pode ser calculada integrando o histograma da distri-
buição de massa, esse resultado nos indica a massa total observada do aglomerado aberto,
como foi observado em trabalhos discutidos na seção 1. Contudo, estamos dedicados a
estimar uma parcela de massa relacionada às estrelas não observadas, isto é, as estrelas
que possuem baixa luminosidade e que não foram detectadas pelo Gaia ou as estrelas de
magnitudes maior que 19, as quais foram retiradas devido ao limite de incompletude do
Gaia.
Deste modo, para estimar a parcela de massa das estrelas não observadas, usamos
o método matemático de extrapolação. Para isso, tomamos como base as funções de
massa das estrelas observadas e extrapolamos até um limite inferior de massa de 0, 09𝑀⊙,
semelhante ao trabalho de (Bonatto; Bica, 2005). Já para o limite superior de integração,
esse corresponde ao valor de massa da estrela mais massiva da amostra.
Para exemplificar a extrapolação da região não observada de estrelas de baixa
massa, geramos um aglomerado sintético com uma amostra de 1000 estrelas a uma dis-
tância de 0.5 kpc, além dos parâmetros fundamentais conforme a figura abaixo.


A extrapolação, linha pontilhada em verde do painel direito da figura 25, segue a
mesma inclinação da função de massa observada (função de massa teórica) e se inicia no
último valor do intervalo de massa teórica, estendendo-se até limite inferior.
Devido à qualidade atual dos dados GAIA EDR3 contribuindo para uma me-
lhor definição dos membros dos aglomerados abertos, conseguimos determinar funções de
massa para estrelas individuais, primárias e secundárias dos aglomerados reais. Esse fato
nos levou a testar dois métodos de integração para determinar as massas totais:
• FM integrada: é a determinação da massa total integrando a função de massa com
todas as estrelas observadas somado ao valor de massa extrapolada, referente às
estrelas não observadas, exatamente como na figura 25.
• FM detalhada: é a integração de cada função de massa observada das estrelas in-
dividuais, primárias e secundárias. Para estas funções de massa, realizamos uma
extrapolação da parcela de massa não observada e somamos seus valores de integra-
ção a fim de obter a massa total final.

Como exemplo do funcionamento da FM detalhada, geramos novamente um aglo-
merado sintético com os mesmo parâmetros fundamentais da figura 25, mas alteramos os
números de estrelas membros para 2000, pois queríamos uma boa definição de intervalos
de massas para a população de estrelas individuais. O resultado está organizado conforme
a figura 26, em que cada painel se encontra uma função de massa, a qual representa a sua
respectiva população − mais detalhes sobre este método será discutido no capítulo 4.

imagem aqui

Por fim, é importante estimar o erro que nosso método comete ao determinar
a massa total de aglomerados abertos usando a FM integrada e a FM detalhada. Essa
estimativa será dada através de um resultado consequente dos testes de recuperação de
massa que serão melhor discutidos no capítulo 3 a seguir.

     ''')
     

