#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:42:22 2021

@author: mrezky
"""

import rebound
import numpy as np
import matplotlib as mpl
from time import time, ctime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Indicates the starting time of this code run
start = ctime(time())
print(start)

sim = rebound.Simulation()
sim.units = ('yr','AU','kg')

plt.ioff()
mpl.use('Agg')

au = 1.495978707e11             # astronomical units, in m
y = 365.25*24*3600              # year-to-second conversion

#mass calculation from density
def mass(rho, r):
    return (4/3) * np.pi * (r**3) * rho

#density calculation from radius
def dens(m, r):
    return 3 * m / (4 * np.pi * r**3)

#collisional cross-section of a particle
def sigm(r):
    return np.pi * r**2

#initial core/sphere properties
def sphr(a, M, rho, R):
    m = mass(rho, R)                                # total mass of the core, in kg
    r = a * (m/(3*M))**(1/3)                        # Hill's radius, in au
    P = 2*np.pi*np.sqrt(a**3 / (sim.G * M))         # orbital timescale, in yr
    rho_c = dens(m, r*au)                           # sphere density, in kg m-3
    tff = np.sqrt(3*np.pi/(32*6.674e-11*rho_c))/y   # free-fall time, in yr                                   # free-fall time, in yr
    return m, r, tff, P

#initial particle properties
def pari(m, d, s, Nn):
    mpr = mass(rho_p, d)            # mass of each particle in physical, in kg
    N = m/mpr                       # the amount of particles in real system
    sn = s * N / Nn                 # coll. xsection in sim, in m2
    an = np.sqrt(sn/np.pi)          # radius of a particle, in m
    mp = m/Nn                       # mass of a sim particle in kg
    return an, mp, N

#particle properties after contraction
def parc(al, rl, rn):
    an = al * (rn/rl)**(3/2)        # new radius of particle, in m
    dn = dens(m_p, rn*au)           # new core density, in kg m-3
    mn = mass(dn, an)               # new mass of particle, in kg
    return an, mn, dn

#filling factor
def filf(N, a, r):
    return N * a**3 / r**3 

#uniformed random sequencer
def rand_uniform(minimum, maximum):
    return np.random.uniform()*(maximum-minimum)+minimum

#generate particle for each species
def particles(N, m, rp):
    n = 0
    while n < N:
        u = rand_uniform(0, 1)
        ct = rand_uniform(-1, 1)
        r = r_h * u**(1/3)
        theta = np.arccos(ct)
        phi = rand_uniform(0, 2*np.pi)
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)
        vx = rand_uniform(-v_dp, v_dp)
        vy = rand_uniform(-v_dp, v_dp)
        vz = rand_uniform(-v_dp, v_dp)
        p = rebound.Particle(simulation=sim, m=m, r=rp, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        sim.add(p)
        n += 1

#user-defined coefficient of restitution function
#(the param from REBOUND does not working as expected)
def cor_bridges(r, v):
	eps = cor						
	return eps

# STANDARD MODEL INPUT ###################################################
#integrator options
sim.integrator = "ias15"
sim.gravity = "compensated"
sim.dt = 0.0002*2.*np.pi
sim.testparticle_type = 1
sim.ri_ias15.epsilon = 1e-6
sof_mp = 1                      #softening param. multiplier, if necessary
tff_mp = 1.3                    #free-fall time multiplier, sim limit

#collision and boundary options
sim.collision = "direct"
cor = 0.1
sim.coefficient_of_restitution = cor_bridges
sim.collision_resolve = "hardsphere"
sim.track_energy_offset = 1

#core properties
M_star = 1.989e30   # mass of the central star, in kg
r_p = 100000        # radius of the core, in m
rho_p = 1000        # density of the core, in kg m-3
a = 5               # sphere distance from the star, in au

#scaled-particle parameters, for 1 species of particle
N_pl = 1000         # number of particles in sim
v = 0               # maximum velocity dispersion in m/s, 0 = error
af1 = 1             # radius of a particle in real system, in m

##############################################################################

#calculating derived parameters from the inputs
v_dp = v * y / au                               # v converted to au/yr
m_p, r_h, tff, P = sphr(a, M_star, rho_p, r_p)  # finding core physical and dynamical properties
s_cr1 = sigm(af1)                               # particle's collisional xsection in real system, in m2
a1, mp1, Np = pari(m_p, af1, s_cr1, N_pl)       # finding the mass and radius of a particle in the simulation
f = filf(N_pl, a1, r_h*au)                      # filling factor
sim.softening = sof_mp * a1/au                  # based on the a_{c,num}, in au

#particle generating process
np.random.seed(70)
particles(N_pl, mp1, a1/au)

#calculate the initial number of particles, energy, and angular momentum of the system
sim.move_to_com()
smx, smy, smz = sim.calculate_angular_momentum()
am0 = np.sqrt(smx**2 + smy**2 + smz**2)
E0 = sim.calculate_energy()
N0 = sim.N
print("Total particles: ", N0)
print('-----')

#initiate the timestep and empty arrays for data collection
times = np.linspace(0.,tff_mp*tff,100)
Ncol = np.zeros(len(times))
Ncot = np.zeros(len(times))
errors = np.zeros(len(times))
am = np.zeros(len(times))
ams = np.zeros(len(times))
rcm = np.zeros(len(times))
ots = np.zeros(len(times))

rl = r_h
al = a1
ml = mp1
dl = dens(m_p, r_h*au)

#initiate and write the initial parameters of the simulation
file = open("summary_{a}.txt".format(a=start),"w")
file.write("SUMMARY OF THE SIMULATION")
file.write("\n")
file.write("INPUT PARAMETERS \n")
file.write("The mass of the central star: {a} kg \n".format(a=M_star))
file.write("The radius of the core: {a} m \n".format(a=r_p))
file.write("The density of the core: {a} kg m-3 \n".format(a=rho_p))
file.write("Distance between the core and its central star: {b} au \n".format(b=a))
file.write("The maximum velocity dispersion of the particles: {a} m/s \n".format(a=v))
file.write("Radius of the physical particles: {a} m \n".format(a=af1))
file.write("The amount of particles in the simulation: {a} \n".format(a=N_pl))
file.write("The coefficient of restitution: {a} \n".format(a=cor))
file.write("Tolerance: {a} \n".format(a=sim.ri_ias15.epsilon))
file.write("---------- \n")
file.write("CALCULATED PARAMETERS \n")
file.write("Total mass of the core: {a} kg \n".format(a=m_p))
file.write("Free-fall time: {a} years \n".format(a=tff))
file.write("Orbital timescale: {a} years \n".format(a=P))
file.write("This numerical integration was done for {a} of its orbital timescale \n".format(a=round(tff/P,4)))
file.write("Collisional cross-section of the real particle: {a} m2 \n".format(a=s_cr1))
file.write("---------- \n")
file.write("THE STATE OF EACH INTEGRATION STEP \n")

#starting the integration and data collection for each timestep
print('NUMERICAL INTEGRATION IS IN PROGRESS')
for i,t in enumerate(times):
    #numerical integration
    sim.integrate(t)
    errors[i] = (sim.calculate_energy() - E0)/E0
    amx, amy, amz = sim.calculate_angular_momentum()
    Ncol[i] = sim.collisions_Nlog/N0
    if t == 0:
        Ncot[i] = 0
    else:
        Ncot[i] = sim.collisions_Nlog/(N_pl*t*y)
    if am0 == 0:
        am[i] = 0
    else:
        am[i] = (np.sqrt(amx**2 + amy**2 + amz**2) - am0)/am0
    
    print('-----')
    print("Integration time (tff)       : ", t/tff)
    print('Sphere radius (rhill)        : ', rl/r_h)
    print("Core mass (kg)               : ", m_p)
    print("Sphere density (kg m-3)      : ", dl)
    print("Phys. particle radius (m)    : ", af1)
    print("Phys. coll. xsection (m2)    : ", s_cr1)
    print("Num. amt of particle         : ", N_pl)
    print("Num. particle radius (km)    : ", al/1e3)
    print("Initial num. part. rad. (km) : ", a1/1e3)
    print("Num. particle mass (kg)      : ", ml)
    print("Filling factor               : ", f)
    print("Collision occured            : ", sim.collisions_Nlog)
    print("Rel. energy error            : ", errors[i])
    print("Rel. angular momentum error  : ", am[i])
    
    if i % 5 == 0:
        coords = np.zeros((6,sim.N))
        
        vm = 0
        #plotting the position and velocity vector of the particles (2D)
        for j in range(sim.N):
            coords[0][j] = sim.particles[j].x
            coords[1][j] = sim.particles[j].y
            coords[2][j] = sim.particles[j].z
            coords[3][j] = sim.particles[j].vx
            coords[4][j] = sim.particles[j].vy
            coords[5][j] = sim.particles[j].vz
            vp = np.sqrt(coords[3][j]**2+coords[4][j]**2+coords[5][j]**2)
            if vp > vm:
                vm = vp
                
        fig1, ax1 = plt.subplots()
        ax1.quiver(coords[0],coords[1],coords[3]/vm,coords[4]/vm)
        ax1.scatter(coords[0],coords[1], s=4, label=r"$\sigma_{c} = $" + str(round(s_cr1,2)) + r"$ m^2$")
        ax1.set_xlim(-r_h, r_h)
        ax1.set_ylim(-r_h, r_h)
        ax1.axis('equal')
        plt.xlabel('X (AU)')
        plt.ylabel('Y (AU)')
        plt.legend(loc='upper right')
        plt.title(r"$t = $" + str(round(t/tff,4)) + r"$\times t_{ff}$")
        fig1.tight_layout()
        plt.savefig("2d_{a}.png".format(a=i))
        plt.close(fig1)
        
        #plotting the position of the particles (3D)
        fig2 = plt.figure(figsize=(10,10))
        ax2 = plt.axes(projection='3d')
        ax2.scatter3D(coords[0],coords[1],coords[2], s=10, label=r"$\sigma_{c} = $" + str(round(s_cr1,2)) + r"$ m^2$")
        ax2.set_xlabel('X (AU)')
        ax2.set_ylabel('Y (AU)')
        ax2.set_zlabel('Z (AU)')
        ax2.set_xlim(-r_h, r_h)
        ax2.set_ylim(-r_h, r_h)
        ax2.set_zlim(-r_h, r_h)
        plt.title(r"$t = $" + str(round(t/tff,4)) + r"$\times t_{ff}$")
        plt.legend()
        fig2.tight_layout()
        plt.savefig("3d_{a}.png".format(a=i))
        plt.close(fig2)
    
    file.write("\n")
    file.write("Time: {a} of tff \n".format(a=t/tff))
    file.write("Total collision counter: {a} times \n".format(a=sim.collisions_Nlog))
    file.write("Relative energy offset: {a} \n".format(a=errors[i]))
    file.write("Relative angular momentum offset: {a} \n".format(a=am[i]))
    file.write("\n")
    file.write("Cloud sphere radius: {a} times rhill \n".format(a=rl/r_h))
    file.write("Sphere density: {a} kg m-3 \n".format(a=dl))
    file.write("The amount of particles in the physical system: {a} \n".format(a=Np))
    file.write("Radius of a particle: {a} km \n".format(a=al/1e3))
    file.write("Mass of a particle: {a} kg \n".format(a=ml))
    file.write("Softening parameters: {a} km \n".format(a=sim.softening*au/1e3))
    file.write("Filling factor: {a}\n".format(a=f))
    file.write("............ \n")

    if i % 1 == 0:
        ra = 0
        xx = 0
        yy = 0
        xv = 0
        yv = 0
        vv = 0
        ls = 0
        for h in range(sim.N):
            px = sim.particles[h].x             # in au
            py = sim.particles[h].y
            pz = sim.particles[h].z
            vx = sim.particles[h].vx * au/y     # in m/s
            vy = sim.particles[h].vy * au/y
            vz = sim.particles[h].vz * au/y
            xx += px
            yy += py
            xv += vx
            yv += vy
            ra += np.sqrt(px**2+py**2+pz**2)**2
            vv += np.sqrt(vx**2+vy**2+vz**2)**2
        
        xcm = xx/sim.N
        ycm = yy/sim.N 
        xcv = xv/sim.N
        ycv = yv/sim.N 
        
        for h in range(sim.N):
            px = sim.particles[h].x
            py = sim.particles[h].y
            vx = sim.particles[h].vx * au/y
            vy = sim.particles[h].vy * au/y
            xb = px - xcm
            yb = py - ycm
            vxb = vx - xcv
            vyb = vy - ycv
            ls += (xb*au * vyb) - (yb*au * vxb)
        
        ams[i] = ls/sim.N
        rcm[i] = np.sqrt(xcm**2 + ycm**2)
                
        vrms = np.sqrt(vv/sim.N)
        ots[i] = vrms * sigm(2*af1) * Np * 3/(4*np.pi*(rl*au)**3)
        
        rrms = np.sqrt(ra/sim.N)            # radius of the new sphere, in au
        f = filf(N_pl, al, rrms*au)         # filling factor update
        rl = rrms
    
    sim.simulationarchive_snapshot("stmod_novar_{a}.bin".format(a=start))  
    
#plotting the energy offset and collision number of the simulation
fig3 = plt.figure(figsize=(14,10))
axa = plt.subplot(321)
plt.plot(times/tff, errors)
plt.yscale("log")
axa.set_ylabel(r"Rel. $E$ error")
axb = plt.subplot(322)
plt.plot(times/tff, am)
axb.set_ylabel(r"Rel. $l$ error")
axc = plt.subplot(323)
plt.plot(times/tff, Ncol)
plt.yscale("log")
axc.set_ylabel(r"$N_{c,num} / N_0$")
axd = plt.subplot(324)
plt.plot(times/tff, ams)
axd.set_ylabel(r"$l_{z}$")
axd.set_xlabel(r"Time [$t_{ff}$]")
axe = plt.subplot(325)
plt.plot(times/tff, rcm)
axe.set_ylabel(r"$r_{com}$")
axe.set_xlabel(r"Time [$t_{ff}$]")
plt.savefig("stats_{a}.png".format(a=start))
plt.close(fig3)

fig4, ax4 = plt.subplots()
plt.plot(times/tff, ots, label=r"Prediction ($\tau_{c}^{-1}$)")
plt.plot(times/tff, Ncot, label="REBOUND")
plt.yscale("log")
ax4.set_ylabel(r"Collision rate ($s^{-1}$)")
ax4.set_xlabel(r"Time [$t_{ff}$]")
plt.legend()
plt.savefig("collstat_{a}.png".format(a=start))
plt.close(fig4)

coords = np.zeros((6,sim.N))
for j in range(sim.N):
    coords[0][j] = sim.particles[j].x
    coords[1][j] = sim.particles[j].y
    coords[2][j] = sim.particles[j].z
    coords[3][j] = sim.particles[j].vx
    coords[4][j] = sim.particles[j].vy
    coords[5][j] = sim.particles[j].vz
    vp = np.sqrt(coords[3][j]**2+coords[4][j]**2+coords[5][j]**2)
    if vp > vm:
        vm = vp

#plotting the final position and velocity vector of the particles (2D)
fig5, ax5 = plt.subplots()
ax5.quiver(coords[0],coords[1],coords[3]/vm,coords[4]/vm)
ax5.scatter(coords[0],coords[1], s=4, label=r"$\sigma_{c} = $" + str(round(s_cr1,2)) + r"$ m^2$")
ax5.set_xlim(-r_h, r_h)
ax5.set_ylim(-r_h, r_h)
plt.xlabel('X (AU)')
plt.ylabel('Y (AU)')
ax5.axis('equal')
plt.legend(loc='upper right')
plt.title(r"$t = $" + str(round(t/tff,4)) + r"$\times t_{ff}$")
fig5.tight_layout()
plt.savefig("2d_end.png")
plt.close(fig5)

#plotting the final position of the particles (3D)
fig6 = plt.figure(figsize=(10,10))
ax6 = plt.axes(projection='3d')
ax6.scatter3D(coords[0],coords[1],coords[2], s=10, label=r"$\sigma_{c} = $" + str(round(s_cr1,2)) + r"$ m^2$")
ax6.set_xlabel('X (AU)')
ax6.set_ylabel('Y (AU)')
ax6.set_zlabel('Z (AU)')
ax6.set_xlim(-r_h, r_h)
ax6.set_ylim(-r_h, r_h)
ax6.set_zlim(-r_h, r_h)
plt.legend()
plt.title(r"$t = $" + str(round(t/tff,4)) + r"$\times t_{ff}$")
fig6.tight_layout()
plt.savefig("3d_end.png")
plt.close(fig6)

print('-----')
print('NUMERICAL INTEGRATION IS FINISHED')
print("-----")

#Indicates the finish time of this code run
finish = ctime(time())
print(finish)

file.write("There are {a} collisions happened during the simulation \n".format(a=sim.collisions_Nlog))
file.write("---------- \n")
file.write("The program started on {a} \n".format(a=start))
file.write("and ended on {a} \n".format(a=finish))
file.close()