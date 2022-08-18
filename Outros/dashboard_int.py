# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:15:51 2022

@author: Anderson Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from oc_tools_padova_edr3 import *
from astropy import units as u
from astropy.coordinates import SkyCoord 
# import seaborn as sns
# from astropy import units as u       #importações necessárias 
# from astropy.coordinates import SkyCoord
# from astropy.coordinates import  FK4
# import astropy.coordinates as coord
# from scipy import optimize
# from scipy import stats


cluster_name = 'Alessi_3'


#lendo isócronas
diretorio = r'S:\Área de Trabalho\software'
grid_dir = (diretorio + '\clusters\gaia_dr3\grids\\')
mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='GAIA_eDR3')
filters = ['Gmag','G_BPmag','G_RPmag']
refMag = 'Gmag' #lendo do grid de isócronas a referencia de magnitude absoluta


#parametros fundamentais
cluster = pd.read_csv(r'S:\Área de Trabalho\software\results_eDR3_likelihood_2022_ptbr\results\log-results-eDR3-MF_detalhada.csv',
                              sep=';')

cluster = cluster.to_records()
#----------------------------------------------------------------------------------------------------            
#aplicando o filtro de aglomerados bons
filtro1 = pd.read_csv(r'S:\Área de Trabalho\software\avaliar_ocs\lista_OCs_classificados.csv', sep=';')
filtro = filtro1.to_records()  
ab, a_ind, b_ind = np.intersect1d(cluster['name'],filtro['clusters_bons'],  return_indices=True)
cluster = cluster[a_ind]


# def load_cluster(cluster):
    
###############################################################################
# read memberships
members_ship = pd.read_csv(r'S:\Área de Trabalho\software\results_eDR3_likelihood_2022\membership_data_edr3\{}_data_stars.csv'.
                           format(cluster_name), sep=';')

RA = members_ship['RA_ICRS']
e_RA = members_ship['e_RA_ICRS']

DEC = members_ship['DE_ICRS']
e_DEC = members_ship['e_DE_ICRS']

# mass = members_ship['mass']
# e_mass = members_ship['er_mass']

# comp_mass = members_ship['comp_mass']
# e_comp_mass = members_ship['er_comp_mass']


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

###############################################################################	
# individuais
alpha_high_ind = cluster['alpha_high_ind'][ind]
alpha_high_ind_error = cluster['alpha_high_ind_error'][ind]


alpha_low_ind = cluster['alpha_low_ind'][ind]
alpha_low_ind_error	 = cluster['alpha_low_ind_error'][ind]
	
Mc_ind = cluster['Mc_ind'][ind]
Mc_ind_error = cluster['Mc_ind_error'][ind]

offset_ind = cluster['offset_ind'][ind]
offset_ind_error = cluster['offset_ind_error'][ind]

###############################################################################	
# primarias
alpha_high_prim = cluster['alpha_high_prim'][ind]
alpha_high_prim_error = cluster['alpha_high_prim_error'][ind]

alpha_low_prim = cluster['alpha_low_prim'][ind]
alpha_low_prim_error = cluster['alpha_low_prim_error'][ind]

Mc_prim = cluster['Mc_prim'][ind]
Mc_prim_error= cluster['Mc_prim_error'][ind]

offset_prim = cluster['offset_prim'][ind]
offset_prim_error = cluster['offset_prim_error'][ind]

###############################################################################	
# secundarias
alpha_high_sec = cluster['alpha_high_sec'][ind]
alpha_high_sec_error = cluster['alpha_high_sec_error'][ind]


alpha_low_sec = cluster['alpha_low_sec'][ind]
alpha_low_sec_error = cluster['alpha_low_sec_error'][ind]

Mc_sec = cluster['Mc_sec'][ind]
Mc_sec_error = cluster['Mc_sec_error'][ind]

offset_sec = cluster['offset_sec'][ind]
offset_sec_error = cluster['offset_sec_error'][ind]	

###############################################################################	
# binarias
alpha_high_prim_sec	= cluster['alpha_high_prim_sec'][ind]
alpha_high_prim_sec_error = cluster['alpha_high_prim_sec_error'][ind]


alpha_low_prim_sec = cluster['alpha_low_prim_sec'][ind]
alpha_low_prim_sec_error = cluster['alpha_low_prim_sec_error'][ind]

Mc_prim_sec = cluster['Mc_prim_sec'][ind]
Mc_prim_sec_error = cluster['Mc_prim_sec_error'][ind]

offset_prim_sec = cluster['offset_prim_sec'][ind]
offset_prim_sec_error = cluster['offset_prim_sec_error'][ind]
	
###############################################################################	
# massa total
vis_mass_total = cluster['vis_mass_total'][ind]
vis_mass_total_error = cluster['vis_mass_total_error'][ind]
	
mass_total = cluster['mass_total'][ind]
mass_total_error = cluster['mass_total_error'][ind]

###############################################################################	
# fracao de binarias
bin_frac = cluster['bin_frac'][ind]

print(bin_frac)


###############################################################################	
#CMD
#isocrona
grid_iso = get_iso_from_grid(age,(10.**FeH)*0.0152,filters,refMag, nointerp=False)
fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True) 
cor_obs = members_ship['BPmag']-members_ship['RPmag']
absMag_obs = members_ship['Gmag']

plt.figure()
# a massa aqui é a massa do sistema
ind = np.argsort(members_ship['mass'] + members_ship['comp_mass'])
#ind_bin = np.where(mod_cluster_obs['bin_flag'] == 1) #sinalizando as binárias
plt.scatter(cor_obs,absMag_obs, c = members_ship['mass'], cmap='jet_r', s = 10)

plt.title('{} \n log(age) = {} $\pm {}$ ; Dist = {} $\pm {}$ ; Av = {} $\pm {}$ ; FeH = {} $\pm {}$ '.format(cluster_name, 
                                                                            age[0], e_age[0],
                                                                            dist[0], e_dist[0],
                                                                            Av[0], e_Av[0],
                                                                            FeH[0], e_FeH[0]))

plt.plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag'], 'r',alpha=0.2)
clb = plt.colorbar(label='$M\odot$')
clb.ax.set_title('M$_\odot$')
plt.ylabel(r'$G_{mag}$')
plt.xlabel(r'$G_{BP}-G_{RP}$')
plt.grid(alpha = 0.2)
plt.ylim(20,5)


###############################################################################	   
# RA x DEC 
# a massa é organizada de acordo com a massa da primaria
ind = np.argsort(members_ship['mass'])
plt.figure()
plt.scatter(members_ship['RA_ICRS'][ind], members_ship['DE_ICRS'][ind],c= members_ship['mass'][ind],cmap='jet_r')
plt.title('{}'.format(cluster_name))
plt.ylabel('RA')
plt.xlabel('DEC')
clb = plt.colorbar()
clb.ax.set_title('M$_\odot$')



###############################################################################	
# Segregação de massa

c1 = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=dist*u.kpc) #centro aglomerado
c2 = SkyCoord(ra=members_ship['RA_ICRS']*u.degree, dec=members_ship['DE_ICRS']*u.degree, distance=dist*u.kpc) #distancia angular estrelas

star_dist = np.array(c1.separation_3d(c2)*1000)

plt.figure()
Mc = 1.0
plt.suptitle('Segregação de massa')
mass_members_ship = members_ship['mass'] + members_ship['comp_mass'] 
plt.hist(star_dist[mass_members_ship < Mc], bins='auto', label='$M_c < 1 M_{\odot} $', alpha=0.4, density = True) #nao consigo incluir as estrelas binarias pq nao temos RA e DEC delas
plt.hist(star_dist[mass_members_ship > Mc], bins='auto', label='$M_c > 1 M_{\odot}$', alpha=0.4, density = True)
count, bins = np.histogram(star_dist[mass_members_ship < Mc], bins='auto',density = True)
count01, bins01 = np.histogram(star_dist[mass_members_ship > Mc], bins='auto', density = True)
plt.vlines(x=np.average(star_dist[mass_members_ship < Mc]), 
           ymin=0, ymax=count.max(), ls='--', lw=2, colors='blue', alpha=0.6)
plt.vlines(x=np.average(star_dist[mass_members_ship > Mc]), 
           ymin=0, ymax=count01.max(), ls='--', lw=2, colors='orange', alpha=0.8)
plt.xlabel('pc')
plt.ylabel('Contagem')
plt.legend()



# load_cluster(cluster)















































