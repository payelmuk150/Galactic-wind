# We solve fluid equations for finding solutions for a supersonic Galactic wind.
# We take into account gravitational potential, magnetic field, cosmic rays and Alfven waves.
# We ignore Galactic rotation.

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve

#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
# I will use natural units h_bar = k_b = c = 1

f = open('Galactic_Wind_Cosmic_Rays_sonic_start_in.txt', 'w')
f1 = open('outflow_speed.txt', 'w')
f2 = open('field_compare.txt', 'w')

G = 7.56395e+15  # in MeV^-1
dr_pc = 10 # 100 pc, 0.1 kpc
R_in = 1000
R_far = 9e+4

cm_to_MeV_inv = 5.0e+10
pc_to_cm = 3.086e+18
erg_to_MeV = 624151.0
cm_inv_to_MeV = 2.0e-11

P_IGM = 1e-15 * erg_to_MeV * cm_inv_to_MeV**3    # Boundary Pressure at 500 kpc

R_termination_pc = 2.0e+5
R_termination = R_termination_pc * pc_to_cm * cm_to_MeV_inv

R_sonic = 200.0 * 1000 * pc_to_cm * cm_to_MeV_inv
index = int( ((R_sonic / (pc_to_cm * cm_to_MeV_inv)) - R_in) / dr_pc )

r_pc = np.arange(R_in, (R_sonic / (pc_to_cm * cm_to_MeV_inv)) + dr_pc, dr_pc) # z_grid_in_pc
r_pc_sonic_onwards = np.arange(R_sonic / (pc_to_cm * cm_to_MeV_inv), R_far + dr_pc, dr_pc) # radius from sonic to far boundary

#R_term = 1e+12 * pc_to_cm * cm_to_MeV_inv # termination shock radius
R_in_nat = R_in * pc_to_cm * cm_to_MeV_inv

r_nat = r_pc * pc_to_cm * cm_to_MeV_inv # in MeV^-1
r_out_nat = r_pc_sonic_onwards * pc_to_cm * cm_to_MeV_inv
dr_pc_nat = dr_pc * pc_to_cm * cm_to_MeV_inv # in MeV^-1
#print(r_nat)

gamma = 4.0 / 3 # EOS index_for_CRs_4.0 / 3
gamma_ref = gamma

eps_B = 0.1
rad_initial = np.arange(0, R_in, dr_pc)

def gravity(r):

	#M_gal = 8e+10 # solar mass_Milky_Way_mass_baryonic
	v0 = (180 / 3e+5)
	rc = 5 * 1000 * pc_to_cm * cm_to_MeV_inv
	rvir = 500 * 1000 * pc_to_cm * cm_to_MeV_inv
	R0 = 8.0 * 1000 * pc_to_cm * cm_to_MeV_inv # Solar position
	Mvir = 1e+12

	# Bulge parameters
	M1 = 2.05e+10 # Bulge
	a1 = 0.0
	b1 = 0.495 * 1000 * pc_to_cm * cm_to_MeV_inv

	# Disk parameters
	M2 = 9.0e+10
	a2 = 7.258 * 1000 * pc_to_cm * cm_to_MeV_inv
	b2 = 0.520 * 1000 * pc_to_cm * cm_to_MeV_inv

	halo_contrib = (+ r * v0**2 / (r**2 + R0**2 + rc**2))

	bulge_contrib = G * M1 * (a1 + np.sqrt(r**2 + b1**2)) * r / ((r**2 + b1**2)**0.5 * (R0**2 + (a1 + np.sqrt(r**2 + b1**2))**2)**1.5) 
	disk_contrib = G * M2 * (a2 + np.sqrt(r**2 + b2**2)) * r / ((r**2 + b2**2)**0.5 * (R0**2 + (a2 + np.sqrt(r**2 + b2**2))**2)**1.5) 
	
	return halo_contrib + bulge_contrib + disk_contrib


def grav_potential(r):
	#M_gal = 8e+10 # solar mass_Milky_Way_mass_baryonic
	v0 = (180 / 3e+5)
	rc = 5 * 1000 * pc_to_cm * cm_to_MeV_inv
	rvir = 500 * 1000 * pc_to_cm * cm_to_MeV_inv
	R0 = 8.0 * 1000 * pc_to_cm * cm_to_MeV_inv # Solar position
	Mvir = 1e+12

	# Bulge parameters
	M1 = 2.05e+10 # Bulge
	a1 = 0.0
	b1 = 0.495 * 1000 * pc_to_cm * cm_to_MeV_inv

	# Disk parameters
	M2 = 9.0e+10
	a2 = 7.258 * 1000 * pc_to_cm * cm_to_MeV_inv
	b2 = 0.520 * 1000 * pc_to_cm * cm_to_MeV_inv

	potential_halo = (v0**2 / 2.0) * np.log(1 + ((r**2.0 + R0**2.0) / rc**2.0)) 
			
	potential_disk = - G * M1 / (R0**2 + (a1 + (r**2 + b1**2)**0.5)**2)**0.5
	potential_bulge = - G * M2 / (R0**2 + (a2 + (r**2 + b2**2)**0.5)**2)**0.5

	potential_halo_vir = np.log(1 + ( (rvir**2.0 + R0**2.0)  / rc**2.0)) * (v0**2 / 2.0)

	potential_disk_vir = - G * M1 / (R0**2 + (a1 + (rvir**2 + b1**2)**0.5)**2)**0.5
	potential_bulge_vir = - G * M2 / (R0**2 + (a2 + (rvir**2 + b2**2)**0.5)**2)**0.5


	return (potential_halo + potential_disk + potential_bulge) - (potential_halo_vir + potential_disk_vir + potential_bulge_vir)

index_area = 2.0
R0 = 8.0 * 1000 * pc_to_cm * cm_to_MeV_inv # Solar position
rs = 20.0 * 1000 * pc_to_cm * cm_to_MeV_inv
A0 = 2 * np.pi * 2.0 * 8.0 * (1000 * pc_to_cm * cm_to_MeV_inv)**2
print('grav_potential_base', grav_potential(R_in_nat))

def area(r):

	Wind_area = A0 * (1 + (r / rs)**index_area)

	return Wind_area

#print('rs', rs)
#print('area1',  ( (R_in_nat / rs)**2.0) )
#print('area2', ((R_in_nat * 200 / rs)**2.0) )

r_total = np.concatenate([r_nat, r_out_nat])
plt.plot(r_total / (1000 * pc_to_cm * cm_to_MeV_inv), (2 * np.abs(grav_potential(r_total)))**0.5 * 3e+5, linewidth = 2 )

plt.ylabel(' Galactic escape speed (km s$^{-1}$)', fontsize = 14)
plt.xlabel('Distance from the disk (kpc)', fontsize = 14)
plt.tick_params(axis='both', labelsize=14)
#plt.title('Milky Way Dark matter Escape Speed')
plt.grid()
plt.show()

#plt.plot(r_total / (1000 * pc_to_cm * cm_to_MeV_inv),  np.abs(grav_potential(r_total)))
plt.semilogy(r_total / (1000 * pc_to_cm * cm_to_MeV_inv),  gravity(r_total) * 3e+10 / 6.58e-22 )

plt.ylabel('Gravitational acceleration (cm/s^2)')
plt.xlabel('Radius (kpc)')
#plt.title('Milky Way Dark matter Escape Speed')
plt.grid()
plt.show()

plt.semilogy(r_total / (1000 * pc_to_cm * cm_to_MeV_inv),  np.abs(grav_potential(r_total)) )

plt.ylabel('Gravitational potential (nat units)')
plt.xlabel('Radius (kpc)')
#plt.title('Milky Way Dark matter Escape Speed')
plt.grid()
plt.show()

B0 = 2.0e-6
B = B0

m_wind = 570 # ~ 0.5 GeV mass of wind particle
#m_wind = 1000 # ~ 0.5 GeV mass of wind particle
#rho_cm3 = 4.156e-4
rho_cm3 = 2.5e-3
rho_start = rho_cm3 * m_wind / (cm_to_MeV_inv**3)
rho = rho_start
P_start = 2e-13 * 624151 / (cm_to_MeV_inv**3)
P_c = P_start

T = 1e+5
P_start_gas = rho_cm3 * T * 1.38e-16 * 624151 / (cm_to_MeV_inv**3) 
print('starting gas pressure', rho_cm3 * T * 1.38e-16)
gamma_gas = 5.0 / 3
P_g = P_start_gas

P_wave_start = 4.0e-16 * 5 * 624151 / (cm_to_MeV_inv**3) 
P_wave = P_wave_start
deltaB = (P_wave * 8 * np.pi / (erg_to_MeV * cm_inv_to_MeV**3))**0.5
print('Turbulent B', (P_wave * 8 * np.pi/ (erg_to_MeV * cm_inv_to_MeV**3))**0.5)

v_alf = (2.18e+11 / 3e+10) * (B) * (m_wind / 1000)**-0.5 * rho_cm3**-0.5

M_dot = 0.02198 / (1.78e-27 * 1.52e+21 / (3.17e-8 * 2e+33)) # reference case_0.0pgas_ 22kpc_scale_2e-13_500pcstart_dB/B~0.5_4.0/3_index_rho02.5e-3

v = M_dot / (rho * area(R_in_nat))
M_alf = v / v_alf

cs_sq = ((gamma_gas * P_start_gas / rho) + (gamma * P_start / rho) * (M_alf + 0.5)**2 / (M_alf + 1)**2
+ (P_wave / rho) * (3 * M_alf + 1) / (2 * (M_alf + 1)))

print('sound_speed_base', cs_sq**0.5 * 3e+5)
print('alfven speed', v_alf * 3e+5)
print('Alfven Mach', M_alf)


print('Inner Boundary CR Pressure (erg/cc)', P_start /(erg_to_MeV * cm_inv_to_MeV**3))
print('Inner Boundary gas Pressure (erg/cc)', P_g /(erg_to_MeV * cm_inv_to_MeV**3))


print('Inner Boundary mass density (gm/cc)', rho *  1.78e-27 * (5e+10)**3 )
print('Mdot', M_dot * 1.78e-27 * 1.52e+21 / (3.17e-8 * 2e+33))
i = 0

Bernoulli = ((v**2 / 2) + ((gamma_gas / (gamma_gas - 1)) * P_g / rho) + grav_potential(R_in_nat) + ((gamma / (gamma - 1)) * (P_c / rho) * (v + v_alf) / v)
			+ (P_wave / rho) * (3 * v + 2 * v_alf) / v )

#Bernoulli = (v**2 / 2) + grav_potential(R_in_nat) + ((gamma / (gamma - 1)) * (P_c / rho) * (v + v_alf) / v)
print('Bernoulli_base', Bernoulli)

consistency = Bernoulli * rho * v * 3e+10 * 1.6e-6 * (5e+10)**3 # erg/cm^3 * cm/s
observed = 8e+40 / (3.14 * 17**2 * (3.086e+21)**2) # erg/(s. cm^2)

#print('loss rate unit area', loss_rate_unit_area)
print('consistency', consistency)
print('observed', observed)

for rad in r_nat :

	rho_cm3 = rho * (cm_to_MeV_inv**3) / m_wind
	Area_r = area(rad)
	B = B0 * area(R_in_nat) / Area_r
	deltaB = (P_wave * 8 * np.pi / (erg_to_MeV * cm_inv_to_MeV**3))**0.5

	v_alf = (2.18e+11 / 3e+10) * (B) * (m_wind / 1000)**-0.5 * rho_cm3**-0.5

	M_alf = v / v_alf
	cs_sq = (gamma_gas * P_g / rho) + ((gamma * P_c / rho) * (M_alf + 0.5)**2 / (M_alf + 1)**2) + ((P_wave / rho) * (3 * M_alf + 1) / (2 * (M_alf + 1)))


	scale_height = area(rad)**-1 * A0 * index_area * rad**(index_area - 1) / rs**index_area
	numerator = cs_sq * scale_height - gravity(rad)

	denominator = v**2 - cs_sq
	RAM = rho * v**2

	dv = v * numerator * dr_pc_nat / denominator
	B_delta = np.sqrt(0.5 * 0.2 * rho * v**2 * 1.602e-6 * (5.06e+10)**3 * 8.0 * np.pi)  # Gauss

	drho = - rho * scale_height * dr_pc_nat - (rho / v) * dv 
	dP_c = (gamma * P_c / rho) * (2 * v + v_alf) * drho / (2 * v + 2 * v_alf)
	dP_gas = gamma_gas * P_g * drho / rho
	dP_wave = (1 / (2 * (M_alf + 1))) * ((3 * M_alf + 1) * (P_wave / rho) * drho - dP_c)

	Bernoulli = ((v**2 / 2) + ((gamma_gas / (gamma_gas - 1)) * P_g / rho) + grav_potential(rad) + ((gamma / (gamma - 1)) * (P_c / rho) * (v + v_alf) / v)
		+ (P_wave / rho) * (3 * v + 2 * v_alf) / v)

	P_g += dP_gas
	P_c += dP_c
	rho += drho
	P_wave += dP_wave
	v += dv
	rad_kpc = rad / (pc_to_cm * cm_to_MeV_inv * 1000)

	f.write(f'{rad / (pc_to_cm * cm_to_MeV_inv)} {v * 3e+5} {cs_sq**0.5 * 3e+5} {v_alf * 3e+5} {B_delta} {rho * (cm_to_MeV_inv**3) / m_wind} {P_g / (erg_to_MeV * cm_inv_to_MeV**3)} {P_c / (erg_to_MeV * cm_inv_to_MeV**3)} {P_wave / (erg_to_MeV * cm_inv_to_MeV**3)} {RAM / (erg_to_MeV * cm_inv_to_MeV**3)} {B}\n')
	f1.write(f'{rad / (pc_to_cm * cm_to_MeV_inv)} {(v + v_alf) * 3e+5} {B} {(P_wave * 8 * np.pi / (erg_to_MeV * cm_inv_to_MeV**3))**0.5} {B_delta}\n')

	i += 1

print('Far pressure', P_c / (erg_to_MeV * cm_inv_to_MeV**3) )

print(v * 3e+5)
f.close()
f1.close()

f1 = np.loadtxt('Galactic_Wind_Cosmic_Rays_sonic_start_in.txt')
plt.semilogx(f1[:,0] / 1000, f1[:,1], label = 'Gas speed')
plt.semilogx(f1[:,0] / 1000, f1[:,2], label = 'Composite sound speed')
plt.semilogx(f1[:,0] / 1000, f1[:,3], label = 'Alfven speed')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel('Height from the disk (kpc)', fontsize = 14)
plt.ylabel('Speed (km s$^{-1}$)', fontsize = 14)
plt.grid()
plt.tight_layout()
plt.show()

#plt.semilogy(f1[:,0], f1[:,6], label = 'P$_g$')
plt.semilogy(f1[:,0] / 1000, f1[:,7], label = 'P$_c$')
plt.semilogy(f1[:,0] / 1000, f1[:,6], label = 'P$_g$')
plt.semilogy(f1[:,0] / 1000, f1[:,8], label = 'P$_w$')
plt.semilogy(f1[:,0] / 1000, f1[:,9], label = 'P$_{ram}$ = $\\rho v^2$')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.loglog(f1[:,0], f1[:,10]**2 / (8 * np.pi), label = 'B^2 / (8 pi)')
plt.xlabel('Height from the disk (kpc)', fontsize = 14)
plt.ylabel('Pressure (erg cm$^{-3}$)', fontsize = 14)
plt.grid()
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()

plt.loglog(f1[:,0] / 1000, f1[:,10] / 1e-6 , label = 'Regular magnetic field')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.semilogx(f1[:,0] / 1000, (f1[:,8] * 8 * np.pi)**0.5 , label = 'Turbulent magnetic field')

plt.ylabel(r'Mean magnetic field ($\rm \mu G$)', fontsize = 14)
plt.xlabel('Height from the disk (kpc)', fontsize = 14)
plt.grid()
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()

plt.loglog(f1[:,0] / 1000, f1[:,5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Height from the disk (kpc)', fontsize = 14)
plt.ylabel('Number density (cm$^{-3}$)', fontsize = 14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid()
plt.show()
