
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:14:19 2020

@author: agez

M2 : test autour du M2 facteur de qualité
v1.0 : avec import PVB_source
v1.1 : mise en forme des parser
v1.2 : import source MA + Nov
"""
import PVB_Source as pvb
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams['figure.constrained_layout.use'] = False
import argparse
import time


start_time = time.time()

parser = argparse.ArgumentParser()  
# General
parser.add_argument('-resolution',  type=int,     default = 20,   help='resolution')
parser.add_argument('-sxy',         type=float,   default = 7,    help='transverse size of the cell (without PML)')
parser.add_argument('-sz',          type=float,   default = 2,    help='longitudinal size of the cell (without PML)')
parser.add_argument('-dpml',        type=float,   default = 0.9,  help='PML thickness')
parser.add_argument('-runtime',     type=float,   default = 10,   help='run until')

parser.add_argument('-beam',        type=str,     default ='rad_ma', help='type de faisceau : plane,gauss,azi_ma,rad_ma or azi_nov rad_nov')
parser.add_argument('-wvl',         type=float,   default = 500,  help='en nm, longueur d onde')
parser.add_argument('-NA',          type=float,   default = 0.5 , help='Numerical aperture')

# For Paraxial Laguerre-Gauss method
parser.add_argument('-M2',          type=float,   default = 1.5 ,   help='M² quality factor (only for MA method)')

# For Non-Paraxial Novotny method
parser.add_argument('-w0',          type=float,   default = 1000 ,help='in um, beam waist before aplanatic lens (only for Novotny method)')
parser.add_argument('-f',           type=float,   default = 1000 ,help='in um, focal length of the aplanatic lens (only for Novotny method)')
parser.add_argument('-n',           type=float,   default = 1,    help='refractive index of the source medium')

args = parser.parse_args()

#%%% PARAMETERS %%%

# General
resolution  = args.resolution
sx          = args.sxy
sy          = args.sxy
sz          = args.sz
dpml        = args.dpml
runtime     = args.runtime
beam        = args.beam
wvl         = args.wvl/1000
NA          = args.NA 

# For Paraxial Laguerre-Gauss method
M2          = args.M2     

# For Non-Paraxial Novotny method
w0          = args.w0
f           = args.f   
n           = args.n 

# Calculated parameters
fcen        = 1/wvl                     # source central frequency
k           = 2*n*np.pi*fcen            # source central wavenumber
waist       = wvl/(np.pi*NA)            # waist at focal point (MA method)

# Output Name
beamtxt     = beam+'_M2='+str(M2)+'_Sxy='+str(sx)+'_Sz='+str(sz)+'_reso='+str(resolution)+'_Wvl='+str(wvl)+'_NA='+str(NA)+'_t='+str(runtime)

#%%% CELL PROPERTIES %%%

cell_x=sx+2*dpml
cell_y=sy+2*dpml
cell_z=sz+2*dpml
cell_size = mp.Vector3(cell_x,cell_y,cell_z)

pml_layers = [mp.Absorber(dpml)]   


#%%% SOURCE PROPERTIES %%%
center_x = 0
center_y = 0
center_z = - sz/2
source_size = mp.Vector3(cell_x  ,cell_y  ,0.0)
source_pos  = mp.Vector3(center_x,center_y,center_z)

Source_Gauss_MA = [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ex,
                             amp_func = lambda r: pvb.LG(r.x,r.y,center_z,0,fcen,waist,M2))]
 

Source_Azi_MA =   [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ex,
                             amp_func = lambda r: pvb.PVB_ma_x(r.x,r.y,center_z,fcen,waist,M2)),
              
                   mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ey,
                             amp_func = lambda r: pvb.PVB_ma_y(r.x,r.y,center_z,fcen,waist,M2))]

Source_Rad_MA =   [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Hx,
                             amp_func = lambda r: pvb.PVB_ma_x(r.x,r.y,center_z,fcen,waist,M2)),
              
                   mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Hy,
                             amp_func = lambda r: pvb.PVB_ma_y(r.x,r.y,center_z,fcen,waist,M2))]

Source_Azi_Nov = [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ex,
                             amp_func = lambda r: pvb.PVB_nov_x(r.x,r.y,center_z,k,n,NA,f,w0)),
                  
                  mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ey,
                             amp_func = lambda r: pvb.PVB_nov_y(r.x,r.y,center_z,k,n,NA,f,w0))]

Source_Rad_Nov = [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Hx,
                             amplitude = -1,
                             amp_func = lambda r: pvb.PVB_nov_x(r.x,r.y,center_z,k,n,NA,f,w0)),
                  
                  mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Hy,
                             amplitude = -1,
                             amp_func = lambda r: pvb.PVB_nov_y(r.x,r.y,center_z,k,n,NA,f,w0))]

Source_Gauss_Nov = [mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ex,
                             amp_func = lambda r: pvb.Ex_H00(r.x,r.y,center_z,k,n,NA,f,w0)),
                    
                    mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ey,
                             amp_func = lambda r: pvb.Ey_H00(r.x,r.y,center_z,k,n,NA,f,w0)),
                    mp.Source(mp.ContinuousSource(frequency=fcen,is_integrated = True),
                             center = source_pos,
                             size = source_size,
                             component = mp.Ez,
                             amp_func = lambda r: pvb.Ez_H00(r.x,r.y,center_z,k,n,NA,f,w0))]

if beam == 'gauss_ma' :
    Source = Source_Gauss_MA
    symmetries = [mp.Mirror(mp.Y,+1),mp.Mirror(mp.X,-1)]
if beam == 'azi_ma' :
    Source = Source_Azi_MA
    symmetries = [mp.Mirror(mp.Y,-1),mp.Mirror(mp.X,-1)]
if beam == 'rad_ma' :
    Source = Source_Rad_MA   
    symmetries = [mp.Mirror(mp.Y,+1),mp.Mirror(mp.X,+1)]
if beam == 'gauss_nov' :
    Source = Source_Gauss_Nov
    symmetries = [mp.Mirror(mp.Y,+1),mp.Mirror(mp.X,-1)]    
if beam == 'azi_nov' :
    Source = Source_Azi_Nov
    symmetries = [mp.Mirror(mp.Y,-1),mp.Mirror(mp.X,-1)]
if beam == 'rad_nov' :
    Source = Source_Rad_Nov 
    symmetries = [mp.Mirror(mp.Y,+1),mp.Mirror(mp.X,+1)]
    
#%%% RUN %%%

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    sources=Source,
                    # geometry=geometry,
                    symmetries=symmetries,
                    force_complex_fields = True,
                    boundary_layers = pml_layers)

sim.run(until=runtime)

#%% DATA COLLECTION

    # Visualization box

boxX = 4*wvl
boxY = 4*wvl
boxZ = sz
box=mp.Volume(center=mp.Vector3(0,0,0),size=mp.Vector3(boxX,boxY,boxZ))
(x,y,z,w)= sim.get_array_metadata(vol=box)


    #Coupe longitudinale (X,Z)
monitor_longi=mp.Volume(center=mp.Vector3(0,0,0),size=mp.Vector3(boxX,0,boxZ))

ex_longi = sim.get_array(vol=monitor_longi, component=mp.Ex)
ey_longi = sim.get_array(vol=monitor_longi, component=mp.Ey)
ez_longi = sim.get_array(vol=monitor_longi, component=mp.Ez)
hx_longi = sim.get_array(vol=monitor_longi, component=mp.Hx)
hy_longi = sim.get_array(vol=monitor_longi, component=mp.Hy)
hz_longi = sim.get_array(vol=monitor_longi, component=mp.Hz)
 
iz0      = int(boxZ*resolution/2)
Iline    = abs(ex_longi[:,iz0])**2+abs(ey_longi[:,iz0])**2+abs(ez_longi[:,iz0])**2
Ie2      = np.amax(Iline)*0.135335283
Ie2line  = np.full(len(x), Ie2)
Hline    = abs(hx_longi[:,iz0])**2+abs(hy_longi[:,iz0])**2+abs(hz_longi[:,iz0])**2

# Plan transverse transmis (X,Z)

monitor_XY=mp.Volume(center=mp.Vector3(0,0,0),size=mp.Vector3(boxX,boxY,0))

ex = sim.get_array(vol=monitor_XY, component=mp.Ex)
ey = sim.get_array(vol=monitor_XY, component=mp.Ey)
ez = sim.get_array(vol=monitor_XY, component=mp.Ez)
hx = sim.get_array(vol=monitor_XY, component=mp.Hx)
hy = sim.get_array(vol=monitor_XY, component=mp.Hy)
hz = sim.get_array(vol=monitor_XY, component=mp.Hz)

I0  = abs(ex)**2
I90 = abs(ey)**2
I45 = 0.5*abs(ex+ey)**2
IPC = 0.5*abs(ex+1j*ey)**2
   
S0 = I0+I90
S1 = I0-I90
S2 = 2*I45-S0
S3 = S0-2*IPC

Psi = 0.5*np.arctan2(S2,S1)
Chi = 0.5*np.arcsin(S3/S0)
    

#%%% CROSS SECTION PLOTS %%%

extentXY = [x[0], x[len(x)-1], y[0], y[len(x)-1]]
extentXZ = [x[0], x[len(x)-1], z[0], z[len(z)-1]]

# Plot of E²(x,y) and H²(x,y)

plt.figure()
ax = plt.subplot(121)
ax.set_title('E²')
plt.imshow(np.transpose(abs(ex)**2+abs(ey)**2+abs(ez)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)')  
ax.set_ylabel('y (\u03BCm)') 
plt.colorbar()

ax = plt.subplot(122)
ax.set_title('H²')
plt.imshow(np.transpose(abs(hx)**2+abs(hy)**2+abs(hz)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)')  
plt.colorbar()

plt.savefig(beamtxt+'_E2_H2_wo.png',dpi=300)
plt.show()
plt.close()

# Plot of E²(x,z) and H²(x,z) + profiles E²(x,z=0) and H²(x,z=0)

plt.figure()
ax = plt.subplot(221)
ax.set_title('E² longitudinal')
plt.imshow(np.transpose(abs(ex_longi)**2+abs(ey_longi)**2+abs(ez_longi)**2),cmap='gray',extent=extentXZ,origin='lower')
ax.set_aspect('equal')
ax.set_ylabel('z (\u03BCm)') 
plt.colorbar()

ax = plt.subplot(223)
ax.plot(x,Iline)
if beam == 'gauss_ma' :
    ax.plot(x,Ie2line,linestyle='dashed',linewidth=1)
ax.set_xlabel('x (\u03BCm)')  
ax.set_ylabel('E²(z=0)') 
plt.grid(True)

ax = plt.subplot(222)
ax.set_title('H² longitudinal')
plt.imshow(np.transpose(abs(hx_longi)**2+abs(hy_longi)**2+abs(hz_longi)**2),cmap='gray',extent=extentXZ,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)')  
plt.colorbar()

ax = plt.subplot(224)
ax.plot(x,Hline)
ax.set_xlabel('x (\u03BCm)') 
plt.grid(True) 

plt.savefig(beamtxt+'_Longitudinal.png',dpi=300)
plt.show()
plt.close()  

# Plot of component Ex²,Ey²,Ez² and Hx², Hy³, Hz²
  
plt.figure()
ax = plt.subplot(2,3,1)
ax.set_title('Ex²')
plt.imshow(np.transpose(abs(ex)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_ylabel('y (\u03BCm)') 
plt.colorbar()

ax = plt.subplot(2,3,2)
ax.set_title('Ey²')
plt.imshow(np.transpose(abs(ey)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
plt.colorbar()

ax = plt.subplot(2,3,3)
ax.set_title('Ez²')
plt.imshow(np.transpose(abs(ez)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
plt.colorbar()

ax = plt.subplot(2,3,4)
ax.set_title('Hx²')
plt.imshow(np.transpose(abs(hx)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_ylabel('y (\u03BCm)') 
ax.set_xlabel('x (\u03BCm)') 
plt.colorbar()

ax = plt.subplot(2,3,5)
ax.set_title('Hy²')
plt.imshow(np.transpose(abs(hy)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)') 
plt.colorbar()

ax = plt.subplot(2,3,6)
ax.set_title('Hz²')
plt.imshow(np.transpose(abs(hz)**2),cmap='gray',extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)') 
plt.colorbar()

plt.savefig(beamtxt+'_Component.png',dpi=300)
plt.show()
plt.close()  
 
# Plot of Stokes coeff

plt.figure()
ax = plt.subplot(1,2,1)
ax.set_title('Psi')
plt.imshow(np.transpose(Psi),extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)')  
ax.set_ylabel('y (\u03BCm)')  
plt.colorbar()

ax = plt.subplot(1,2,2)
ax.set_title('Chi')
plt.imshow(np.transpose(Chi),vmin=-np.pi/4,vmax=np.pi/4,extent=extentXY,origin='lower')
ax.set_aspect('equal')
ax.set_xlabel('x (\u03BCm)')  
plt.colorbar()

plt.savefig(beamtxt+'_Stokes.png',dpi=300)
plt.show()
plt.close()


#%% Display running time
print("\nTemps d execution : %s secondes ---" % (time.time() - start_time))    
         
