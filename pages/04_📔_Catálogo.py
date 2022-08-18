# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:27:24 2022

@author: Anderson Almeida
"""

import streamlit as st
from PIL import Image

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