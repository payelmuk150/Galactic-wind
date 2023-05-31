import numpy as np 
import sys
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.special as sc
from scipy import interpolate
#np.set_printoptions(threshold=sys.maxsize)

f = open('NoDamping_wind_proton_thesis_opt.txt', 'w')
f_Noemie = open('Outgoing_spectrum_300kpc_H_opt.txt', 'w')
f_disk = open('disk_spectrum_H.txt', 'w')
f_multi = open('multimessenger.txt', 'w')
f_CR = np.loadtxt('CR_all_particle.txt')
f_CR_1 = np.genfromtxt('proton_ATIC2.txt', delimiter = ',')
f_CR_H = np.genfromtxt('proton-KASCADE.txt', delimiter = ',')
f_CR_2 = np.genfromtxt('protons-Icetop.txt', delimiter = ' ')
element = 'proton'

f1 = np.loadtxt('transport_onlygas.txt')
f_velocity = np.loadtxt('transport_exp_10percent_power.txt')
f_vel = np.loadtxt('Galactic_Wind_Cosmic_Rays_sonic_start_in.txt')

u_gas = f_vel[:,1]

A = 1.0
Z = 1.0

p_inj = 1.0e+0 * A # GeV
p_min = 1.03e+3
log_pmin = np.log10(p_min)
log_pinj = np.log10(p_inj)

ps = 10**(np.linspace(log_pmin, 8.6 , 120))
p_short = 10**(np.linspace(log_pinj, log_pmin, 50))
print(ps)
n = 10

momentum_bins_min = ps[0:-4]
momentum_bins_max = ps[4:]

momentum_bins_min_1 = ps[0:-4]
momentum_bins_max_1 = ps[4:]

momenta = ps #np.sqrt(momentum_bins_min_1 * momentum_bins_max_1)
#print(momenta[80])
dps = momentum_bins_max - momentum_bins_min
dps_loss = ps[1:] - ps[0:-1]

u = 1000 # km/s
sigma = 3.5
q = 3.0 * sigma / (sigma - 1.0) # power index 
M_dot = 2.0 # solar mass per year_mass_loading_Bustard
P_IGM = 1e-14 # IGM pressure erg/cm^3
T_07 = (u / 0.129)**(1.0 / 0.527)

R_b = 500 # kpc outer boundary
km_to_kpc = 3.2408e-17
cm_to_kpc = 3.2408e-22
#u_kpc = u * km_to_kpc * np.tanh(radii / 20.0) # wind speed in kpc/s

R_in = f1[0,0] # inner boundary kpc
radii = f1[:,0] # radii less and equal to R_shock in kpc
R_shock = f1[-1,0]
dr = np.zeros(len(radii))

n_cc = f1[:, 4]
dr_short = radii[1:] - radii[:-1]
dr[1:] = dr_short
dr[0] = dr[1] # dr in kpc

print('Shock radius kpc : ', R_shock)

print(len(radii))
xi = radii / R_shock
xi_in = R_in / R_shock
d_xi = dr / R_shock
length_p = len(momenta)
length_r = len(radii)

Area = np.zeros((length_r, length_p))
u_kpc = np.zeros((length_r, length_p))
erf = np.zeros((length_r, length_p))

for i in range(len(u_kpc[0,:])):
	u_kpc[:,i] = f_velocity[:,1] * km_to_kpc / 1.0

for i in range(len(u_kpc[0,:])):
	Rd = 10.0 # disk radius in kpc
	A0 = np.pi * Rd**2.0
	zs = 20.0 # scale height radius of flux tube
	Area[:,i] = A0 * (1 + (radii / zs)**2)


for i in range(len(u_kpc[0,:])):
	z0 = 0.1 # kpc
	erf[:,i] = sc.erf(radii / (z0 * np.sqrt(2)))

#print(Area)
#print(u_kpc / km_to_kpc)

u_kpc_norm = u_kpc[-1,0]
radius_1 = np.arange(R_shock, R_b + 1.0, 1.0)
u_beyond = np.zeros(len(radius_1))
u_beyond[0] = u_kpc_norm / km_to_kpc
u_beyond[1:] = (u_kpc_norm / sigma) * (R_shock / radius_1[1:])**2.0 / km_to_kpc

Area_downstream = np.zeros((len(radius_1), length_p))

for i in range(len(u_kpc[0,:])):
	Rd = 10.0 # disk radius in kpc
	A0 = np.pi * Rd**2.0
	zs = 20.0 # scale height radius of flux tube
	Area_downstream[:,i] = A0 * (1 + (radius_1 / zs)**2)

n_cc_downstream = (f1[-1, 4] * 3.5) * np.ones(len(radius_1))

#u_plot = ( (u_kpc[:,0] / km_to_kpc) * np.heaviside(R_shock - radius_1, 0)
#			+ (u_kpc_norm / 4.0) * np.heaviside(radius_1 - R_shock,0) * (R_shock / radius_1)**2.0)
#print(radius_1)
plt.semilogx(radii, u_kpc / km_to_kpc, 'b', linewidth = 2 )
plt.semilogx(radius_1, u_beyond, 'b', linewidth = 2)
plt.grid()
plt.xlim([0.1, 500])
plt.xlabel('z (kpc)', fontsize = 14)
plt.ylabel('Wind speed (km/s)', fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.show()

grid = np.zeros((length_r, length_p))
xi_2D = np.zeros((length_r, length_p))

D2grid = np.zeros((len(radius_1), length_p))
D1grid = np.zeros((length_r, length_p))
alpha_2_grid = np.zeros((length_r, length_p))
#source =  np.zeros((length_r, length_p))
q_tild_k =  np.zeros((length_r, length_p))
source_reacc =  np.zeros(length_p)
G10 = np.zeros((length_r, length_p))
source = np.zeros((length_r, length_p))

#index = 0.42
norm_D = 2.0e+28
norm_D_down = 1e+28
norm_D_down_down = 1e+25
index_r = 0.0
delta = 0.4

coherence_length = radii * 1000 / 30 # pc
coherence_length_cm = coherence_length * 3.08e+18 # pc

coherence_length_down =  1000 # pc
coherence_length_down_cm = coherence_length_down * 3.08e+18 # pc

#larmor_radius = 3.3e+4 * 3.24e+17 * (momenta / Z)# pc
B_microG = (f1[:,2]**2)**0.5
turb_upstream = (B_microG / (f1[:,2] * 0.1) )**2

B_insanity = (f1[-1,5]**2 * 1.5)**0.5
eta_B = f1[:,5]**2 / (f1[:,2]**2)
#eta_B = np.clip(eta_B, -1e+10, 1)
print('etaB', eta_B)
turbulence_delta = 1.6667
turb_index = (turbulence_delta - 1.0)

larmor_radius_pc = 3.3e+4 * 3.24e-17 * (momenta / Z) / (B_microG[-1] / 1e+6)
print('larmor_radius_pc', larmor_radius_pc[80], ps[80])

#H = 0.88, He = 0.09, Ox = 0.0031, Carbon = 0.0031, Mg = 0.00066, Si = 0.00066, Ne = 0.0048, Fe = 6.3e-4,  CNO = 0.0063, SiNeMg = 0.0017

R_d = 10.0  # disk radius in kpc
SN_rate = 1.6 / (100 * 3.15 * 10**7)
E_SN = 1e+51 # erg
frac_CR = 0.08
frac_element = 0.88
m_particle = 1.0 # GeV
eta = 1.0 / 1.0

I_P_norm = SN_rate * E_SN * frac_CR * frac_element * 624 * delta
#print('source_norm' , I_P_norm)
p_gal = 3.0e+6

Rd = 10.0 # disk radius in kpc
I_P = I_P_norm * p_inj**(delta) / ( 2 * 4.0 * np.pi * np.pi * R_d**2.0)
i_50 = 0

for i, _ in enumerate(radii):
	larmor_radius_pc = 3.3e+4 * 3.24e-17 * (momenta / Z) / (B_microG[i] / 1e+6)
	larmor_radius_cm = 3.3e+6 * (momenta / Z) / (B_microG[i] / 1e+6)
	c = 3.0e+10 # speed of light cm/s
	#D1grid[i,:] = norm_D * (momenta / Z)**index * (radii[i] / R_shock)**index_r * cm_to_kpc**2.0 # power law diffusion
	#D1grid[i,:] = 3.13e+23 * ps * cm_to_kpc**2.0

	if(radii[i] <= 1.0):
		D1grid[i,:] = norm_D * (momenta / Z)**0.3 * cm_to_kpc**2.0

	elif(radii[i] > 1.0 and radii[i] < 40.0):
		index = larmor_radius_pc < 1000 * 1e+10
		index1 = larmor_radius_pc >= 1000 * 1e+10
		i_50 = i

		D1grid[i,index] = (1.0 / 3) * (((coherence_length_cm[i] * c * turb_upstream[i]
		* (coherence_length[i] / larmor_radius_pc[index])**(turbulence_delta - 2.0) * cm_to_kpc**2.0) / (2 * 3.14)**(2 / 3))
		+ (coherence_length_cm[i] * turb_upstream[i] * c * (larmor_radius_pc[index] / coherence_length[i])**2.0 * cm_to_kpc**2.0 * (2 * 3.14 * 2 / 3)))

	elif(radii[i] >= 40.0 and radii[i] < 100.0):
		index = larmor_radius_pc < 1000 * 1e+10
		index1 = larmor_radius_pc >= 1000 * 1e+10
		larmor_radius_pc_down = 3.3e+4 * 3.24e-17 * (momenta / Z) / (B_insanity / 1e+6)
		larmor_radius_cm_down = 3.3e+6 * (momenta / Z) / (B_insanity / 1e+6)

		D1grid[i,index] = (D1grid[i_50, index] * (1 / (1 + 5 * np.exp((radii[i] - 85.0) / 3) )) + 
		(((1.0 / (3 * (2 * 3.14)**(2 / 3)) ) * coherence_length_down_cm * c 
		* (coherence_length_down / larmor_radius_pc_down[index])**(turbulence_delta - 2.0) * cm_to_kpc**2.0)
		+ (2 * 3.14 * 2 / (3 * 3)) * coherence_length_down_cm * c * (larmor_radius_pc_down[index] / coherence_length_down)**2.0 * cm_to_kpc**2.0))


	elif(radii[i] >= 100.0):
		index = larmor_radius_pc < 1000 * 1e+10
		index1 = larmor_radius_pc >= 1000 * 1e+10
		larmor_radius_pc = 3.3e+4 * 3.24e-17 * (momenta / Z) / (B_insanity / 1e+6)
		larmor_radius_cm = 3.3e+6 * (momenta / Z) / (B_insanity / 1e+6)

		D1grid[i,index] = (((1.0 / (3 * (2 * 3.14)**(2 / 3)) ) * coherence_length_down_cm * c 
		* (coherence_length_down / larmor_radius_pc[index])**(turbulence_delta - 2.0) * cm_to_kpc**2.0)
		+ (2 * 3.14 * 2 / (3 * 3)) * coherence_length_down_cm * c * (larmor_radius_pc[index] / coherence_length_down)**2.0 * cm_to_kpc**2.0)

#print('galactic diffusion coefficient', D1grid[:,-1] / (cm_to_kpc**2.0) )
plt.semilogy(radii, D1grid[:,0] / (cm_to_kpc**2.0), label = 'Diffusion coefficient at 1000 GeV')
plt.xlabel('Distance from Disk (kpc)')
plt.ylabel(f'Diff Coefficient (cm$^2$ / s)')
plt.legend()
plt.show()

larmor_radius_pc = 3.3e+4 * 3.24e-17 * (momenta / Z) / (B_insanity / 1e+6)
larmor_radius_cm = 3.3e+6 * (momenta / Z) / (B_insanity / 1e+6)

plt.loglog(momenta, D1grid[-1,:] / (cm_to_kpc**2.0), label = 'Diffusion coefficient at shock')
#plt.loglog(momenta, (1 / 3) * larmor_radius_cm * 3e+10  * (1 / eta_B[i]) )
plt.xlabel('momentum (GeV)')
plt.ylabel(f'Diff Coefficient (cm$^2$ / s)')
plt.legend()
plt.show()

for i, _ in enumerate(radius_1):
	B_turb = B_insanity * 10
	larmor_radius_pc = 3.3e+4 * 3.24e-17 * (momenta / Z) / (B_turb / 1e+6)
	larmor_radius_cm = 3.3e+6 * (momenta / Z) / (B_turb / 1e+6)
	index = larmor_radius_pc < 1000 * 1e+10
	index1 = larmor_radius_pc >= 1000 * 1e+10

	D2grid[i,index] =  ((((1.0 / (3 * (2 * 3.14)**(2 / 3)) ) * coherence_length_down_cm * c 
		* (coherence_length_down / larmor_radius_pc[index])**(turbulence_delta - 2.0) * cm_to_kpc**2.0)
		+ (2 * 3.14 * 2 / (3 * 3)) * coherence_length_down_cm * c * (larmor_radius_pc[index] / coherence_length_down)**2.0 * cm_to_kpc**2.0) / 1)

plt.loglog(momenta, D1grid[-1,:] / cm_to_kpc**2)
#plt.loglog(momenta, 5e+23 * (momenta / Z)**1.0)
plt.xlabel('Distance from Disk (kpc)')
plt.ylabel(f'Diff Coefficient (cm$^2$ / s)')
plt.title(f'Spatial Dependence of D for p =  {ps[0]:.0f} GeV')
#plt.xlim([1e-3, 198])
plt.grid()
plt.show()

for i , _ in enumerate(radii):
	xi_2D[i,:] = xi[i]

alpha_2_Rb = (u_kpc_norm / sigma) * R_shock * (1.0 - ( R_shock / R_b)) / D2grid[-1,:]

def Gamma2(p):
	Gamma2_values = np.zeros(len(p))
	integrand = (3.0 / (sigma - 1)) / (p * (np.exp(alpha_2_Rb) - 1.0))

	for k in range(len(p)):
		Gamma2_values[k] = np.trapz(integrand[: k + 1], p[: k + 1])

	Gamma2_values = np.clip(Gamma2_values, 1e-10, 1e+10)

	return Gamma2_values

print('Gamma2', Gamma2(ps))

Area_shock = A0 * (1.0 + (R_shock / zs)**2)
Area_start = A0 * (1. + (R_in / zs)**2)

def reacc_source(p, Gamma1, Gamma21):
	p0 = p_inj * 1
	s = 4.0
	delta_s = q - s
	index_reacc = s + delta_s

	p_min_low = p_min
	#print(p_min_low)
	reacc_source_high = np.zeros(len(p))
	reacc_low_energy_contrib = eta * ((index_reacc * p**-index_reacc * I_P * Area_start * np.exp(- Gamma21 - Gamma1) * (1.0 / (1.0 * ((delta - delta_s)) ))
			                   / (Area_shock * u_kpc_norm)) * (+ p0**-(delta - delta_s) - p_min_low**-(delta - delta_s)))
	#print('reacc_low_energy_contrib', reacc_low_energy_contrib)

	integrand = p**(index_reacc - 1.0) * p**(- q - delta) * np.exp(Gamma1 + Gamma21) * I_P * Area_start * np.exp(- p / (p_gal * Z))
	i = 0

	for k in p:
		reacc_source_high[i] = np.trapz(integrand[:i + 1], p[:i + 1]) 
		i += 1
    
	reacc_source_high = eta * reacc_source_high * index_reacc * p**-index_reacc * np.exp(- Gamma21 - Gamma1) / (Area_shock * u_kpc_norm)

	#print(reacc_low_energy_contrib * np.ones(len(ps)))
	return  reacc_low_energy_contrib + reacc_source_high 
	#print()

for i , _ in enumerate(radii):
	source[i,:] = (momenta)**(-4.0 - delta) * I_P * Area_start * np.exp(- momenta / (p_gal * Z)) # GeV^{-3} s^{-1}

#print('Gamma2', Gamma2(ps))

def G1(xi, p, f, q_tild):
	G1_output = np.zeros((len(xi), len(p)))
	u_gradient = np.gradient(u_kpc[:,0], xi)
	area_gradient = np.gradient(Area[:,0], xi)

	integrand = (1.0 / 3) * ((area_gradient * u_kpc[:,0] + 
	(Area[:,0] * u_gradient)) * (q_tild * f).T).T

	for k in range(len(xi)):

		G1_output[k, :] = np.trapz(integrand[0: k + 1, :], xi[0: k + 1], axis = 0)

	return G1_output

def zeroth_fs(p):
	#return reacc_source(p, 0.0, Gamma2(momenta))
	gamma1 = np.zeros(len(p))
	return reacc_source(p, gamma1, Gamma2(p))

fs0_init = zeroth_fs(momenta)

plt.ylim([1e40,1e+55])
#plt.loglog(momenta, f10_init[0,:] * momenta**4.0, label = f'{R_in} kpc')
#plt.loglog(momenta, f10_init[30,:] * momenta**4.0, label = '80 kpc')
plt.loglog(momenta, fs0_init * momenta**4.0, label = f'R_shock = {R_shock} kpc')
plt.title('Initial guess function : Plane Shock Solution')
plt.grid()
plt.ylabel('p$^4$ f(p)')
plt.xlabel('Energy (GeV)')
plt.legend()

plt.show()


CR_pressure = np.trapz(zeroth_fs(momenta) * 4.0 * A * np.pi * ps[:]**3.0 * 0.00160218  / 3.0, ps[:]) / (3.086e+21)**3.0
#np.savetxt('f2.txt', zeroth_fs(momenta) * 4.0 * np.pi * ps[:]**3.0 * 0.00160218 / 3.0)
print(f'guess_Pressure_{R_shock}_kpc_fs_term', CR_pressure)

def  zeroth_f1(xi_2D, xi, p):
	G_start = np.zeros((length_r, length_p))
	f_start = np.ones((length_r, length_p))

	return (zeroth_fs(p) * np.exp(- exp_factor_f1(xi, p, G_start, f_start))) + iteration_central(xi_2D,p,G_start,f_start)

def exp_factor_f1(xi, p , G, f):

	exp_factor_f1_output = np.zeros((len(xi), len(p)))
	integrand =  (G[:,:] * R_shock / ((Area[:,:] * f[:,:]) * D1grid[:,:]) 
				 + (u_kpc * R_shock / D1grid[:,:]))

	for k in range(len(xi)):
		exp_factor_f1_output[k,:] = np.trapz(integrand[k:, :], xi[k :], axis = 0)
	return exp_factor_f1_output

def Gamma1_Rs(p, G, f): 
	Gamma1_values = np.zeros(len(p))
	integrand = np.zeros(len(p))

	for m in range(len(p)):

		integrand[m] = q * G[m] / (u_kpc_norm * Area_shock * p[m] * f[m])
	
	for k in range(len(p)):
		Gamma1_values[k] = np.trapz(integrand[: k + 1], p[: k + 1])

	return Gamma1_values

def iteration_central(xi, p, G, f): 

	central_output = np.zeros((len(xi), len(p)))
	xi_prime = np.copy(xi)
	xi_double_prime = np.copy(xi)

	integrand_1 = R_shock * erf / (Area * D1grid)

	for n in range((np.shape(xi))[0]):
		#print('n', n)
		integrand2 = ((G[n:,:] * R_shock / ((f[n:,:] * Area[n:,:]) * D1grid[n:,:]))
					 + ( (u_kpc[n:,:]) * R_shock / D1grid[n:,:]))
		integrand2_total = np.zeros((len(xi[n:]), len(p)))
		i = 0
		for m in range(n, (np.shape(integrand_1))[0], 1):

			integrand2_total[i,:] = np.exp(- np.trapz(integrand2[:i + 1,:], xi_double_prime[n:m + 1,:], axis = 0))
			i += 1
		central_output[n, :] = np.trapz(integrand_1[n:,:] * integrand2_total, xi_prime[n:], axis = 0) 
	return central_output * source

f10_init = zeroth_f1(xi_2D, xi, momenta)

CR_pressure = np.trapz(f10_init[0,:] * A * 4.0 * np.pi * ps[:]**3.0 * 0.00160218 / 3.0, ps[:]) / (3.086e+21)**3.0
print('guess_Pressure_at_galaxy', CR_pressure)

CR_density_gal = np.trapz(f10_init[0,:] * A * 4.0 * np.pi * ps[:]**2.0, ps[:]) / (3.086e+21)**3.0
#np.savetxt('f2.txt', zeroth_fs(momenta) * 4.0 * np.pi * ps[:]**3.0 * 0.00160218 / 3.0)
print(f'guess_density at galaxy', CR_density_gal)

x = 0
press = np.zeros(len(radii))

for hello in radii:
	press[x] = np.trapz(f10_init[x,:] * A * 4.0 * np.pi * ps[:]**3.0 * 0.00160218 / 3.0, ps[:]) / (3.086e+21)**3.0
	x += 1

plt.loglog(radii, press, label = 'radial dependence of pressure')
plt.ylabel('Pressure (erg/cc)')
plt.xlabel('z')
plt.legend()
plt.grid()
plt.show()

x = 0
den = np.zeros(len(radii))

for hello in radii:
	den[x] = np.trapz(f10_init[x,:] * A * 4.0 * np.pi * ps[:]**2.0, ps[:]) / (3.086e+21)**3.0
	x += 1

plt.loglog(radii, den, label = 'radial dependence of CR number density')
plt.ylabel('n (/cc)')
plt.xlabel('z')
plt.legend()
plt.grid()
plt.show()


print('wind_base', radii[30])

CR_pressure = np.trapz(f10_init[90,:] * A * 4.0 * np.pi * ps[:]**3.0 * 0.00160218 / 3.0, ps[:]) / (3.086e+21)**3.0
print('guess_Pressure_at_wind_base', CR_pressure)
#print(source_reacc)

#print(np.shape(f10_init))
plt.ylim([1e40,1e+55])
plt.loglog(momenta, f10_init[0,:] * momenta**4.0, label = f'{R_in} kpc, guess function at galaxy')
#plt.loglog(momenta, f10_init[30,:] * momenta**4.0, label = '80 kpc')
plt.loglog(momenta, fs0_init * momenta**4.0, label = f'R_shock = {R_shock} kpc, guess function at shock')
plt.title('Initial guess function : Plane Shock Solution')
plt.grid()
plt.ylabel('p$^4$ f(p)')
plt.xlabel('Energy (GeV)')
plt.legend()

plt.show()

fs0 = np.copy(fs0_init)
f10 = np.copy(f10_init)

upstream1 = momenta**5.0 * f10[0,:] * (3e+10 / (3.08e+21 * 3.08e+19**2))
fs0_init_flux = momenta**5.0 * fs0 * (3e+10 / (3.08e+21 * 3.08e+19**2))

plt.loglog(f_CR[:,0], f_CR[:,1], 'o', label = 'all-particle measured spectrum')
plt.loglog(f_CR_1[:,0], f_CR_1[:,1], 'o', label = f'{element} measured spectrum')
plt.ylabel(f'Intensity at {R_in} kpc (cm$^{-2}$ GeV$^2$ s$^{-1}$ sr$^{-1}$)')

plt.loglog(momenta, upstream1 * A / 1e+4, label = 'Guess function at Galaxy z = 0')
#plt.loglog(momenta, fs0_init_flux, label = 'Guess function at Shock')
plt.grid()
plt.ylim([1e-2,1e+4])
plt.show()


#print(fs0)
#print(f10[0,:])
iterable = 0
G1_k = np.copy(G10)
shape_q = np.shape(f10)

#f1 = np.zeros()

f_sh_all = np.copy(fs0_init)
f_sh_all_1 = np.copy(f10_init[-1,:])

length = (len(f10[0,:]))
ratios = np.ones(length)
ratios1 = np.ones(length)

q_tild_stack = np.zeros(shape_q)

while True :
	print('f_galaxy', fs0)
	print('f_galaxy', f10[0,:])

	f_spatial_derivative = ((f10[1:,:] - f10[:-1,:]).T / dr_short).T
	inner_term = D1grid[:-1,:] * (Area[:-1,:] * f_spatial_derivative)
	inner_term_derivative = ((inner_term[1:,:] - inner_term[:-1,:]).T / dr_short[:-1]).T
	diffusion_total = inner_term_derivative 

	adv_term = (Area[:-2,:] * (u_kpc[:-2,:] * f_spatial_derivative[:-1]))

	flux_2 = (- D1grid[:-1,:] * f_spatial_derivative * 0.0) + (9.715e-12 * f10[:-1,:])
	flux_init = (- D1grid[:-1,:] * f_spatial_derivative * 0.0) + (9.715e-12 * f10_init[:-1,:])

	flux_plot_2 = flux_2 *  ps**5.0 / (3.08e+21)**2.0
	flux_plot_init = flux_init *  ps**5.0 / (3.08e+21)**2.0

	iterable += 1
	print(iterable)
	derivative_k = (-f10[:,0:-4] + f10[:,4:]) / dps

	q_tild_k[:,:-4] = - (3.0 + (np.log(f10[:,4: ] / f10[:, :-4]) / np.log(momenta[4:] / momenta[:-4])) )

	q_tild_k[:,-4] = q_tild_k[:,-5]
	q_tild_k[:,-3] = q_tild_k[:,-5]
	q_tild_k[:,-2] = q_tild_k[:,-5]
	q_tild_k[:,-1] = q_tild_k[:,-5]

	q_tild_k1 = np.clip(q_tild_k, - 20, np.Inf)
	q_abs = np.abs(q_tild_k1)

	shape_f = np.shape(f10)
	derivative_loss = np.zeros(shape_f)

	derivative_loss[:,0:-1] = (-f10[:,0:-1] + f10[:, 1:]) / dps_loss

	derivative_loss[:,-1] = derivative_loss[:,-2]

	G1_k = G1(xi, momenta, f10, q_tild_k1) 
	Gamma1_k = Gamma1_Rs(momenta, G1_k[-1,:], fs0)
	Gamma1_k = np.clip(Gamma1_k, - 300, 300 )
	#print('G_ratio', G1_k[-1,:] / fs0)

	
	LHS = (( D1grid[:-1,:] * f_spatial_derivative) * Area[:-1,:]) - ( (u_kpc[:-1,:] * f10[:-1,:]) * Area[:-1,:] * 0)

	RHS =  ( (u_kpc[:-1,:] * f10[:-1,:]) * Area[:-1,:]) + G1_k[:-1,:] - ((source[:-1,:]) * erf[:-1])

	ratio =  np.abs( (LHS / RHS)[5,:])

	print(ratio)

	stack2 = np.vstack((ratios, ratio))
	ratios = stack2
	fs_k = reacc_source(ps, Gamma1_k, Gamma2(ps))

	f1k  = ((fs_k * np.exp(- exp_factor_f1(xi, momenta, G1_k, f10)))
			+ 1.0 * iteration_central(xi_2D, momenta, G1_k, f10)) 

	f_backstream = (fs_k * np.exp(- exp_factor_f1(xi, momenta, G1_k, f10)))
	f_galactic = iteration_central(xi_2D, momenta, G1_k, f10)

	if(np.any(f1k == 0)):
		print(np.argwhere(f1k == 0))
		break

	stack = np.vstack((f_sh_all, fs_k ))
	f_sh_all = stack

	stack1 = np.vstack((f_sh_all_1, f1k[0,:] ))
	f_sh_all_1 = stack1

	stack4 = np.vstack((q_tild_stack, q_abs[2,:] ))
	q_tild_stack = stack4
	
	if np.any(f1k <= 0):
		np.argwhere(np.any(f1k == 0))
		break

	if(iterable == n):
		break

	fs0 = np.copy(fs_k)
	f10 = np.copy(f1k)

for i in range (n) :
	plt.loglog(momenta, stack[i] * momenta**4)

	plt.ylabel('p$^4$ f(p)')
	plt.xlabel('Energy (GeV)')
	plt.grid()
	plt.ylim([1e40,1e+55])

plt.title(f'Convergence at the Shock : {R_shock} kpc')
plt.show()

for i in range (n) :
	plt.loglog(momenta, stack1[i] * momenta**4, label=f'{i}')

	plt.ylabel('p$^4$ f(p)')
	plt.xlabel('Energy (GeV)')
	plt.grid()
	plt.ylim([1e40,1e+55])

plt.title(f'Convergence at the Galaxy Boundary = {R_in} kpc')

plt.legend(prop={'size': 7})
plt.grid()
plt.show()

fs_final_flux = momenta**5.0 * fs_k * (3e+10 / (3.08e+21 * 3.08e+19**2))
plt.loglog(ps, np.abs(flux_plot_2[0,:] * A * 1e+4), label = f'p$^3$ $\\times$ p$^2$ f(p) ({element}), z = 0' )
plt.loglog(ps, np.abs(flux_plot_init[0,:] * A * 1e+4), label = f'Guess function, z = 0' )
plt.loglog(f_CR[:,0], f_CR[:,1] * 1e+4, 'o',  label = 'all-particle measured spectrum')
plt.loglog(f_CR_1[:,0], f_CR_1[:,1] * 1e+4, 'o', label = f'{element} measured spectrum')
plt.loglog(f_CR_H[:,0], f_CR_H[:,1] * 1e+4, 'o', label = f'{element} measured spectrum')

plt.ylim([1e+2,1e+8])
plt.xlim([1e+2, 1e+10])
plt.ylabel('Intensity at 10 kpc (cm$^{-2}$ GeV$^2$ s$^{-1}$ sr$^{-1}$)')
plt.xlabel('Energy (GeV)')
plt.legend()
plt.grid()
plt.title(f'Converged flux at the disk, z = {R_in} kpc')
plt.show()

plt.loglog(ps, np.abs(fs_k * A * momenta**4.0) / 1e+45, linewidth = 2, label = f'$z = 200 $ kpc' )
plt.loglog(ps, np.abs(f_backstream[0,:] * A * momenta**4.0) / 1e+45, '--', linewidth = 2, label = f'Backstreaming, $z = 0$' )
plt.loglog(ps, np.abs(f_galactic[0,:] * A * momenta**4.0) / 1e+45, '-.', linewidth = 2, label = f'Galactic protons, $z = 0$' )
plt.loglog(ps, np.abs(f1k[0,:] * A * momenta**4.0) / 1e+45, 'k:', linewidth = 2, label = f'Total flux, $z = 0$' )
plt.ylim([1e+40 / 1e+45,1e+55 / 1e+45])
plt.xlim([1e+3, 1e+9])

#plt.ylabel('p$^4$ f(p) (GeV kpc$^{-3}$)')
plt.ylabel('p$^4$ f(p) (arbitrary units)', fontsize = 14)
plt.xlabel('Energy (GeV)', fontsize = 14)
plt.tick_params(axis='both', labelsize=14)
plt.legend(fontsize = 14)
plt.grid()
plt.tight_layout()
plt.show()

k = 0
for momentum in ps:
	print(f'{ps[k]} {fs_k[k]}\n')
	k += 1

plt.loglog(momenta, f10[0,:] * momenta**2, label = f'Solution at the Galaxy boundary : 10 kpc')
plt.loglog(momenta, f10[-1,:] * momenta**2, label = f'Solution at the shock : {R_shock} kpc')
plt.title('Upstream D : 1e+26 p$^{0.5}$ r$^{0.1}$ cm$^2$ / s, Downstream D : 3e+23 p cm$^2$ / s')


plt.ylabel('p$^4$ f(p)')
plt.xlabel('Energy (GeV)')
plt.legend()
plt.grid()
plt.show()	

for j in range(1, n + 1):
	plt.loglog(momenta, stack2[j], label = f'{j}')
	plt.ylabel('LHS / RHS')
	plt.xlabel('Momenta (GeV)')

plt.title('Convergence at the Shock front')

plt.legend(prop={'size': 7})
plt.grid()
plt.show()