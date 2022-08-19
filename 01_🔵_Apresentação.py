# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:50:07 2022

@author: Anderson Almeida
"""




import streamlit as st

st.set_page_config(page_title="Apresentação",layout='centered', page_icon='🔵')

st.sidebar.image("images/logo.png", use_column_width=True)

st.title('Determinação da massa de aglomerados abertos utilizando dados do catálogo EDR3')

st.subheader('Dissertação de mestrado defendida por Anderson Almeida, sob orientação do Prof. Dr. Hektor Monteiro.')
st.write('Universidade Federal de Itajubá')

st.write('''
         
         
         Com a disponibilização dos catálogos de dados Gaia e a existência de métodos de ajuste
de isócronas automatizados, o estudo de aglomerados abertos vem passando por grandes
avanços nos últimos anos. Seus parâmetros fundamentais vêm, consequentemente, sendo
estimados em maior escala e com melhor precisão. Entretanto, parâmetros importantes
como as massas totais desses objetos, os detalhes das populações de estrelas individuais
e binárias e a existência de segregação de massa são pouco estudadas de maneira ade quada. 

Nesse contexto, apresentamos um novo método de determinação de massas individuais, 
inclusive de estrelas binárias. Esse método permite estudar a massa
total dos aglomerados abertos, assim como detalhes da população de binárias através de
suas funções de massa. Para validar o método e sua eficiência, utilizamos aglomerados
sintéticos com parâmetros previamente determinados. Com o método validado, aplicamos
o procedimento para aglomerados de um recente catálogo de parâmetros fundamentais
de nosso grupo, obtido do GAIA Early Data Release 3 (EDR3). Os principais resultados 
adquiridos incluem a obtenção da função de massa detalhada para as populações de
estrelas individuais, primárias e secundárias, bem como as massas totais para 900 aglomerados. 


Sendo assim, esse Dashaboard foi desenvolvido com objetivo de divulgar os resultados obtidos em nossa pesquisa, além de fornecer 
uma interface gráfica para o pesquisador/usuário selecionar. 
    ''')
        
st.subheader('A página ainda está em processo de atualização!')

st.write('''

Para eventuais dúvidas, informações ou colaborações, por favor entre em contato pelo email:
    andersonalmeida_sa@outlook.com
    
    ''')