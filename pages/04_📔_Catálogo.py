# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:27:24 2022

@author: Anderson Almeida
"""

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Catálogo",layout='centered', page_icon='📔')

st.write('''


Os dados utilizados neste trabalho foram gerados no formato .csv e existem dois
conjunto de arquivos:
    
🔹 O primeiro nomeado como "log-results-eDR3-MF_detalhada.csv" é cátálogo 
com os parâmetros fundamentais dos aglomerados abertos, jutamente 
das suas massas totais, fração de binárias, inclinações das funções de massas, 
pontos de virada e razão média de segregação pode ser encontrado. 

🔹 O segundo, localizado na pasta "membership_data_edr3" em que ficam abrigados
para cada aglomerado aberto um arquivo com seu nome. Neste arquivo temos em cada
linha as informações sobre cada estrela que compõe o aglomerado aberto. Bem como 
suas massas determinados por nós. Note que a coluna referente a massa da estrela
esta nomeada como "mass" e caso a estrela seja um binárias, a coluna "comp_mass"
vai ter uma massa correspondente a estrela secundária.

Todos estes arquivos podem ser encontradas na pasta "data" do repositório do GitHub, 
utilizado para criar esse Dashboard. Acesse pelo link:
    
https://github.com/ander-son-almeida/dashboard_aglomerados_abertos

    
    ''')