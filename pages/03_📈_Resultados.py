# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:15:51 2022

@author: Anderson Almeida
"""


import numpy as np
import pandas as pd 
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit 
import streamlit as st

from oc_tools_padova_edr3 import *




def twosided_IMF(m, Mc=0., slopeA=0., offsetA=1., slopeB=-1.0):
    res = []
    for mval in m:
        if mval > Mc:
            res.append(slopeA * mval + offsetA)
        else:
            t1 = slopeA * Mc + offsetA
            t2 = slopeB * Mc
            dt = t1-t2
            res.append(slopeB * mval + dt)
    return np.array(res)


def mass_function(mass, title):
    
    # histogram
    #######################################################################
    
    mass = np.log10(mass[mass > 0.])
    
    mass_cnt, mass_bins = np.histogram(mass,bins='auto')
    
    mass_cnt_er = np.sqrt(mass_cnt)
    mass_cnt_er = ((mass_cnt_er/mass_cnt)/2.303)
    
    mass_cnt = np.log10(mass_cnt)

    mass_bin_ctr = mass_bins[:-1] + np.diff(mass_bins)/2
    
    mass_bin_ctr = mass_bin_ctr[mass_cnt >= 0]
    
    mass_cnt_er = mass_cnt_er[mass_cnt >= 0]
    
    mass_cnt = mass_cnt[mass_cnt >= 0]
    
    # ajust
    #######################################################################
    guess = [0.02,-1., 1., 0.]
    
    try:
        
        popt, pcov = curve_fit(twosided_IMF, mass_bin_ctr, mass_cnt, p0=guess, 
                            sigma=mass_cnt_er,max_nfev=1e5,
                            bounds=([-0.1, -3, 0., -3.0], [0.1, 0.0, np.inf, 3.0]),
                                )
        # ERRO
        sigma = np.sqrt(np.diag(pcov))
        
    except:   
        print(cluster_name, 'Falha no ajuste')
        popt = [np.nan, np.nan, np.nan, np.nan]
        sigma = [np.nan, np.nan, np.nan, np.nan]
        
        
    Mc = np.around(popt[0], decimals = 2)
    alpha_high_mass = np.around(popt[1], decimals = 2)
    offset = np.around(popt[2], decimals = 2)
    alpha_low_mass =np.around(popt[3], decimals = 2)
    
    #erro

    Mc_error = np.around(sigma[0], decimals = 2)
    alpha_high_mass_error = np.around(sigma[1], decimals = 2)
    offset_error = np.around(sigma[2], decimals = 2)
    alpha_low_mass_error = np.around(sigma[3], decimals = 2)
    
    return (alpha_high_mass, alpha_low_mass, Mc, offset, Mc_error, alpha_high_mass_error, 
            offset_error, alpha_low_mass_error, mass_bin_ctr, mass_cnt, mass_cnt_er, popt)




#lendo isócronas

# grid_dir = dir('grids')
mod_grid, age_grid, z_grid = load_mod_grid()
filters = ['Gmag','G_BPmag','G_RPmag']
refMag = 'Gmag' 
    

# mod_grid, age_grid, z_grid = load_mod_grid(grid_dir)
filters = ['Gmag','G_BPmag','G_RPmag']
refMag = 'Gmag' 


#parametros fundamentais
cluster = pd.read_csv('dashboard_aglomerados_abertos/data/log-results-eDR3-MF_detalhada.csv', sep=';')

cluster = cluster.to_records()
#----------------------------------------------------------------------------------------------------            
#aplicando o filtro de aglomerados bons
filtro1 = pd.read_csv('dashboard_aglomerados_abertos/filters\lista_OCs_classificados.csv', sep=';')
filtro = filtro1.to_records()  
ab, a_ind, b_ind = np.intersect1d(cluster['name'],filtro['clusters_bons'],  return_indices=True)
cluster = cluster[a_ind]


# cluster_name = cluster['name']
list_clusters = cluster['name']

st.set_page_config(page_title="My App",layout='wide')

# Using object notation
st.sidebar.header(r"$aglomerados \: abertos$")

cluster_name = st.sidebar.selectbox(
    "Aglomerado aberto:",
    (list(list_clusters))
)






# def load_cluster(cluster):
    
###############################################################################
# read memberships
members_ship = pd.read_csv('dashboard_aglomerados_abertos/data/membership_data_edr3/membership_data_edr3/{}_data_stars.csv'.
                           format(cluster_name), sep=';')

RA = members_ship['RA_ICRS']
e_RA = members_ship['e_RA_ICRS']

DEC = members_ship['DE_ICRS']
e_DEC = members_ship['e_DE_ICRS']


###############################################################################
# Parametros fundamentais
	
ind = np.where(cluster['name'] == cluster_name)

RA = cluster['RA_ICRS'][ind]
DEC = cluster['DE_ICRS'][ind]

age = cluster['age'][ind]
e_age = cluster['e_age'][ind]

dist = cluster['dist'][ind]/1000 #kpc
e_dist = cluster['e_dist'][ind]/1000 #kpc

FeH = cluster['FeH'][ind]
e_FeH = cluster['e_FeH'][ind]

Av = cluster['Av'][ind]
e_Av = cluster['e_Av'][ind]
seg_ratio = cluster['segregation_ratio'][ind]

# st.title('Parâmetros fundamentais:')

# st.info(body, *, icon=None)
# st.info('This is a purely informational message', icon="ℹ️")
st.sidebar.subheader("Parâmetros fundamentais:")
st.sidebar.success(r"$\: log(age)={} \pm {} \\ Dist.={}\pm{} \: kpc \\ Av={}\pm{} \: mag \\ \\  FeH={}\pm{}$".format(age[0],e_age[0],
                                                                                  dist[0],e_dist[0], 
                                                                                  Av[0],e_FeH[0],
                                                                                  FeH[0], e_Av[0]))


###############################################################################	
# massa total
vis_mass_total = cluster['vis_mass_total'][ind]
vis_mass_total_error = cluster['vis_mass_total_error'][ind]
	
mass_total = cluster['mass_total'][ind]
mass_total_error = cluster['mass_total_error'][ind]

###############################################################################	
# fracao de binarias
bin_frac = cluster['bin_frac'][ind]

###############################################################################	
st.sidebar.subheader("Resultados:")
st.sidebar.success(r"$\: M_{{T}}={} \pm {} \: M_{{\odot}} \\ M_{{T-obs}}={} \pm {}  \: M_{{\odot}} \\ Fracao \: Binárias={} \\ Razão \: Segregacao={}$".format(int(mass_total[0]),
                                                                                                                              int(mass_total_error[0]),
                                                                                                                              int(vis_mass_total[0]),
                                                                                                                              int(mass_total_error[0]),
                                                                                                                              bin_frac[0],
                                                                                                                              seg_ratio[0]))




#CMD
#isocrona
grid_iso = get_iso_from_grid(age,(10.**FeH)*0.0152,filters,refMag, nointerp=False)
fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True) 
cor_obs = members_ship['BPmag']-members_ship['RPmag']
absMag_obs = members_ship['Gmag']

#--------------------------------------------------------------------------
import plotly.express as px
import plotly.graph_objects as go


# with col1:
cmd_scatter = pd.DataFrame({'G_BPmag - G_RPmag': cor_obs, 'Gmag': absMag_obs, 'Mass': members_ship['mass']})
cmd_iso = pd.DataFrame({'G_BPmag - G_RPmag': fit_iso['G_BPmag']-fit_iso['G_RPmag'], 'Gmag': fit_iso['Gmag']})



fig1 = px.scatter(cmd_scatter, x = 'G_BPmag - G_RPmag', y = 'Gmag',
                  opacity=0.6, color= 'Mass', color_continuous_scale = 'jet_r')

fig2 = px.line(cmd_iso, x = 'G_BPmag - G_RPmag', y = 'Gmag')

fig = go.Figure(data = fig1.data + fig2.data).update_layout(coloraxis=fig1.layout.coloraxis)
fig.update_layout(xaxis_title= 'G_BP - G_RP (mag)',
                  yaxis_title="G (mag)",
                  coloraxis_colorbar=dict(title="M☉"),
                  yaxis_range=[20,5])

#height=800,width=900



###############################################################################	   
# RA x DEC 
# a massa é organizada de acordo com a massa da primaria

ind = np.argsort(members_ship['mass'])

#--------------------------------------------------------------------------
# with col2:
ra_dec = pd.DataFrame({'RA': members_ship['RA_ICRS'][ind], 'DEC': members_ship['DE_ICRS'][ind], 'Mass': members_ship['mass'][ind]})

fig_ra_dec = px.scatter(ra_dec, x = 'RA', y = 'DEC', color= 'Mass', 
                        color_continuous_scale = 'jet_r')




###############################################################################	
# Segregação de massa
Mc = 1.0
c1 = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=dist*u.kpc) #centro aglomerado
c2 = SkyCoord(ra=members_ship['RA_ICRS']*u.degree, dec=members_ship['DE_ICRS']*u.degree, distance=dist*u.kpc) #distancia angular estrelas
mass_members_ship = members_ship['mass'] + members_ship['comp_mass']
star_dist = np.array(c1.separation_3d(c2)*1000)


import plotly.figure_factory as ff

seg1 = pd.DataFrame({'Mc < 1M☉': star_dist[mass_members_ship < Mc]})

seg2 = pd.DataFrame({'Mc > 1M☉': star_dist[mass_members_ship > Mc]})

seg = pd.concat([seg1,seg2], axis=1)

seg = seg.fillna(0)

hist, bin_edges = np.histogram(star_dist[mass_members_ship < Mc], density=True)
hist2, bin_edges2 = np.histogram(star_dist[mass_members_ship > Mc], density=True)

xaxis_max = np.concatenate((bin_edges, bin_edges2), axis=0)
yaxis_max = np.concatenate((hist, hist2), axis=0)

seg = px.histogram(seg, histnorm='probability density', opacity=0.7)
seg.add_vline(x=np.average(star_dist[mass_members_ship < Mc]), line_dash = 'dash', line_color = 'blue')
seg.add_vline(x=np.average(star_dist[mass_members_ship > Mc]), line_dash = 'dash', line_color = 'red')



seg.update_layout(xaxis_title= 'Distância (pc)',
                  legend={'title_text':''},
                  yaxis_title='Contagem',
                  xaxis_range=[1,xaxis_max.max()],
                  yaxis_range=[0,yaxis_max.max()+0.02])





###############################################################################	
# FM indivuals

ind_indv = members_ship['comp_mass'] == 0
mass =  members_ship['mass'][ind_indv]
title = 'Individuais \n 'r'$\alpha_A = {} \pm {}$; $\alpha_B = {} \pm {}$; $M_c = {} \pm {}$'

(alpha_high_mass, alpha_low_mass, Mc, offset, 
 Mc_error, alpha_high_mass_error, offset_error, 
 alpha_low_mass_error, mass_bin_ctr, mass_cnt, 
 mass_cnt_er, popt) = mass_function(mass, title)

xplot = np.linspace(mass_bin_ctr.min(),mass_bin_ctr.max(),1000)

#--------------------------------------------------------------------------

fm_ind = pd.DataFrame({'mass_bin_ctr': mass_bin_ctr, 'mass_cnt': mass_cnt, 'mass_cnt_er': mass_cnt_er})
fm_ind_adj = pd.DataFrame({'xplot': xplot, 'ajuste': twosided_IMF(xplot, *popt)})

plot_ind1 = px.scatter(fm_ind, x="mass_bin_ctr", y="mass_cnt", error_y="mass_cnt_er")
plot_ind2 = px.line(fm_ind_adj, x = 'xplot', y = 'ajuste', color_discrete_sequence = ['orange'])
plot_ind = go.Figure(data = plot_ind1.data + plot_ind2.data)

plot_ind.update_layout(title='Estrelas Individuais', xaxis_title= '$log(M_{\odot})$',
                                            yaxis_title='$\\xi(log(M_{\odot}))$')




###############################################################################	
# FM primarias

ind_bin = members_ship['comp_mass'] > 0
mass =  members_ship['mass'][ind_bin]
title = 'Primárias \n 'r'$\alpha_A = {} \pm {}$; $\alpha_B = {} \pm {}$; $M_c = {} \pm {}$'


(alpha_high_mass, alpha_low_mass, Mc, offset, 
 Mc_error, alpha_high_mass_error, offset_error, 
 alpha_low_mass_error, mass_bin_ctr, mass_cnt, 
 mass_cnt_er, popt) = mass_function(mass, title)


xplot = np.linspace(mass_bin_ctr.min(),mass_bin_ctr.max(),1000)

#------------------------------------------------------------------------------

fm_prim = pd.DataFrame({'mass_bin_ctr': mass_bin_ctr, 'mass_cnt': mass_cnt, 'mass_cnt_er': mass_cnt_er})
fm_prim_adj = pd.DataFrame({'xplot': xplot, 'ajuste': twosided_IMF(xplot, *popt)})

plot_prim1 = px.scatter(fm_prim, x="mass_bin_ctr", y="mass_cnt", error_y="mass_cnt_er")
plot_prim2 = px.line(fm_prim_adj, x = 'xplot', y = 'ajuste', color_discrete_sequence = ['orange'])
plot_prim = go.Figure(data = plot_prim1.data + plot_prim2.data)

plot_prim.update_layout(title='Estrelas Primárias', xaxis_title= '$log(M_{\odot})$',
                                            yaxis_title='$\\xi(log(M_{\odot}))$')



###############################################################################	
# FM Secundarias

ind_bin = members_ship['comp_mass'] > 0
mass =  members_ship['comp_mass'][ind_bin]

title = 'Secundárias \n 'r'$\alpha_A = {} \pm {}$; $\alpha_B = {} \pm {}$; $M_c = {} \pm {}$'

(alpha_high_mass, alpha_low_mass, Mc, offset, 
 Mc_error, alpha_high_mass_error, offset_error, 
 alpha_low_mass_error, mass_bin_ctr, mass_cnt, 
 mass_cnt_er, popt) = mass_function(mass, title)

xplot = np.linspace(mass_bin_ctr.min(),mass_bin_ctr.max(),1000)

#------------------------------------------------------------------------------

fm_sec = pd.DataFrame({'mass_bin_ctr': mass_bin_ctr, 'mass_cnt': mass_cnt, 'mass_cnt_er': mass_cnt_er})
fm_sec_adj = pd.DataFrame({'xplot': xplot, 'ajuste': twosided_IMF(xplot, *popt)})

plot_sec1 = px.scatter(fm_sec, x="mass_bin_ctr", y="mass_cnt", error_y="mass_cnt_er")
plot_sec2 = px.line(fm_sec_adj, x = 'xplot', y = 'ajuste', color_discrete_sequence = ['orange'])
plot_sec = go.Figure(data = plot_sec1.data + plot_sec2.data)

plot_sec.update_layout(title='Estrelas Secundárias', xaxis_title= '$log(M_{\odot})$',
                                            yaxis_title='$\\xi(log(M_{\odot}))$')




###############################################################################	
# FM Binárias

ind_bin = members_ship['comp_mass'] > 0
mass =  np.concatenate((members_ship['mass'][ind_bin], members_ship['comp_mass'][ind_bin]), axis = 0)

title = 'Binárias \n 'r'$\alpha_A = {} \pm {}$; $\alpha_B = {} \pm {}$; $M_c = {} \pm {}$'

(alpha_high_mass, alpha_low_mass, Mc, offset, 
 Mc_error, alpha_high_mass_error, offset_error, 
 alpha_low_mass_error, mass_bin_ctr, mass_cnt, 
 mass_cnt_er, popt) = mass_function(mass, title)

xplot = np.linspace(mass_bin_ctr.min(),mass_bin_ctr.max(),1000)

#------------------------------------------------------------------------------

fm_bin = pd.DataFrame({'mass_bin_ctr': mass_bin_ctr, 'mass_cnt': mass_cnt, 'mass_cnt_er': mass_cnt_er})
fm_bin_adj = pd.DataFrame({'xplot': xplot, 'ajuste': twosided_IMF(xplot, *popt)})

plot_bin1 = px.scatter(fm_bin, x="mass_bin_ctr", y="mass_cnt", error_y="mass_cnt_er")
plot_bin2 = px.line(fm_bin_adj, x = 'xplot', y = 'ajuste', color_discrete_sequence = ['orange'])
plot_bin = go.Figure(data = plot_bin1.data + plot_bin2.data)

plot_bin.update_layout(title=r'$Estrelas Binárias$', xaxis_title= r"$log(M_{\odot})$",
                                            yaxis_title=r"$\xi(log(M_{\odot}))$")





container1 = st.container()
col1, col2, col3 = st.columns(3)



with container1:
    
    
    with col1:
        st.subheader("CMD")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição RA e DEC")
        st.plotly_chart(fig_ra_dec, use_container_width=True)
        
    with col3:
        st.subheader("Segregação de Massa")
        st.plotly_chart(seg, use_container_width=True)


container2 = st.container()
col4, col5 = st.columns(2)

with container2:
    
    st.header("Funções de Massa")
    
    with col4:
        st.write("dfgdfgdfgfd")
        st.plotly_chart(plot_ind, use_container_width=True)
    
    with col5:
        st.write("dfgdfgdfgfd")
        st.plotly_chart(plot_prim, use_container_width=True)
    


container3 = st.container()
col6, col7 = st.columns(2)

with container3:
    
    
    with col6:
        
        st.plotly_chart(plot_sec, use_container_width=True)

    with col7:
        st.plotly_chart(plot_bin, use_container_width=True)



# def convert_df(df):
#      # IMPORTANT: Cache the conversion to prevent computation on every rerun
#      return df.to_csv().encode('utf-8')

# csv = convert_df(my_large_df)

# st.download_button(
#      label="Download data as CSV",
     # data=csv,
     # file_name='large_df.csv',
     # mime='text/csv',
 # )



































