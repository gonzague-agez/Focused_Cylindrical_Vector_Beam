#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:39:18 2021

@author: Gonzague Agez, Martin Montagnac, Arnaud Arbouet, Vincent Paillard
"""
# Import
import numpy as np
import scipy.special as special
from scipy import integrate

# Laguerre-Gauss method
    
def _cart_cyl(x,y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x,y)
    return rho,phi 
     
def _W(z,fcen,waist,M2):
    Wm=waist*M2
    ZR=np.pi*Wm**2*fcen/M2
    return Wm*np.sqrt(1+np.power(z/ZR,2))  
 
def _R_inv(z,fcen,waist,M2):
    Wm=waist*M2
    ZR=np.pi*Wm**2*fcen/M2
    return z/(z**2+ZR**2)
 
def LG(x,y,z,l,fcen,waist,M2):

    k=2*np.pi*fcen
    rho,phi=_cart_cyl(x,y)
    Wm=waist*M2
    Wz=_W(z,fcen,waist,M2)
    r_inv=_R_inv(z,fcen,waist,M2)
    ZR=np.pi*Wm**2*fcen/M2
    
    return   np.sqrt(2/np.pi)*(Wm/Wz)*(np.sqrt(2)*rho/Wz)**abs(l) \
            *np.exp(-rho**2 / Wz**2) \
            *np.exp(1j*l*phi)  \
            *np.exp(0.5*1j*k*r_inv*rho**2) \
            *np.exp(2*1j*np.arctan(z/ ZR))

            

def PVB_LG_x(x,y,z,fcen,waist,M2):

    return ( -1j/2*(LG(x,y,z,1,fcen,waist,M2)+LG(x,y,z,-1,fcen,waist,M2)))

def PVB_LG_y(x,y,z,fcen,waist,M2):

    return (  1/2*(LG(x,y,z,1,fcen,waist,M2)-LG(x,y,z,-1,fcen,waist,M2)))



# Novotny-Hetch non-paraxial method (Novotny, L., & Hecht, B. (2012). Principles of nano-optics. Cambridge university press.)



def _cart_cyl_NH(x,y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return rho,phi 

def _fw(theta, f0, thetamax):
    fres = np.exp(-(np.sin(theta)/f0*np.sin(thetamax))**2)
    return fres 

def _I11_plus_3xI12(x,y,z,k,n,NA,f,w0):  #Iazm=I11+3*I12

    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax))     
    rho, phi = _cart_cyl_NH(x,y)    
  
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                         *np.sin(theta)**2*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                         *np.sin(theta)**2*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 

def PVB_NH_x(x,y,z,k,n,NA,f,w0):
    cn1=1
    cn2=1
    prefactor = 0.5*1j*k*f**2*np.sqrt(cn1/cn2)*np.exp(-1j*k*f)/w0
    rho, phi = _cart_cyl_NH(x,y)
    I_azm = _I11_plus_3xI12(x,y,z,k,n,NA,f,w0)
    azi_x=prefactor * ( 1j*(I_azm)*np.sin(phi) )
    return azi_x

def PVB_NH_y(x,y,z,k,n,NA,f,w0):
    cn1=1
    cn2=1
    prefactor = 0.5*1j*k*f**2*np.sqrt(cn1/cn2)*np.exp(-1j*k*f)/w0
    rho, phi = _cart_cyl_NH(x,y)
    I_azm = _I11_plus_3xI12(x,y,z,k,n,NA,f,w0)
    azi_y=prefactor * ( -1j*(I_azm)*np.cos(phi) )
    return azi_y  

def I00(x,y,z,fcen,n,NA,f,w0):

    k=2*n*np.pi*fcen
    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax)) 
    rho, phi = _cart_cyl_NH(x,y)    
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)*(1+np.cos(theta))*special.j0(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)*(1+np.cos(theta))*special.j0(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 
          
def I01(x,y,z,fcen,n,NA,f,w0):

    k=2*n*np.pi*fcen
    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax))    
    rho, phi = _cart_cyl_NH(x,y)    
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**2*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**2*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 

def I02(x,y,z,fcen,n,NA,f,w0):

    k=2*n*np.pi*fcen
    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax))
    rho, phi = _cart_cyl_NH(x,y)    
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)*(1-np.cos(theta))*special.jv(2,k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)*(1-np.cos(theta))*special.jv(2,k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 


def I10(x,y,z,fcen,n,NA,f,w0):

    k=2*n*np.pi*fcen
    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax))    
    rho, phi = _cart_cyl_NH(x,y)    
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**3*special.j0(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**3*special.j0(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 


def I11(x,y,z,fcen,n,NA,f,w0):

    k=2*n*np.pi*fcen
    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax))     
    rho, phi = _cart_cyl_NH(x,y)    
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**2*(1+3*np.cos(theta))*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**2*(1+3*np.cos(theta))*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 

def I12(x,y,z,k,fcen,n,NA,f,w0):

    k=2*n*np.pi*fcen
    thetamax = np.arcsin(NA/n)
    f0 = w0/(f*np.sin(thetamax))     
    rho, phi = _cart_cyl_NH(x,y)    
    real = integrate.quad(lambda theta: np.real(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**2*(1-np.cos(theta))*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    imag = integrate.quad(lambda theta: np.imag(_fw(theta,f0,thetamax)*np.sqrt(np.cos(theta))
                          *np.sin(theta)**2*(1-np.cos(theta))*special.j1(k*rho*np.sin(theta))*np.exp(1j*k*z*np.cos(theta))), 0, thetamax)
    
    return real[0] + 1j*imag[0] 

# USED FOR GAUSSIAN (00)

def Ex_H00(x,y,z,fcen,n,NA,f,w0):
    k=2*n*np.pi*fcen
    cn1=1
    cn2=1
    prefactor = 0.5*1j*k*f**2*np.sqrt(cn1/cn2)*np.exp(-1j*k*f)/w0
    rho, phi = _cart_cyl_NH(x,y)
    I_00 = I00(x,y,z,k,n,NA,f,w0)
    I_02 = I02(x,y,z,k,n,NA,f,w0)
    H00_x = prefactor * I_00 + I_02*np.cos(2*phi)
    return H00_x

def Ey_H00(x,y,z,fcen,n,NA,f,w0):
    k=2*n*np.pi*fcen
    cn1=1
    cn2=1
    prefactor = 0.5*1j*k*f**2*np.sqrt(cn1/cn2)*np.exp(-1j*k*f)/w0
    rho, phi = _cart_cyl_NH(x,y)
    I_02 = I02(x,y,z,k,n,NA,f,w0)
    H00_y = prefactor * I_02 * np.sin(2*phi)
    return H00_y

def Ez_H00(x,y,z,fcen,n,NA,f,w0):
    k=2*n*np.pi*fcen
    cn1=1
    cn2=1
    prefactor = 0.5*1j*k*f**2*np.sqrt(cn1/cn2)*np.exp(-1j*k*f)/w0
    rho, phi = _cart_cyl_NH(x,y)
    I_01 = I01(x,y,z,k,n,NA,f,w0)
    H00_z = prefactor * (-2 * 1j * I_01 * np.cos(phi) )   
    return H00_z




