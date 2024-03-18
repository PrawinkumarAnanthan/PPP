import os
import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot,mesh
from dolfinx.fem import Function, functionspace,assemble_scalar
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner

def adaptive_timestep(time_stores,r_val):
    print (f"Timestep entering finding dt is {time_stores}")
    print (f"r_val entering finding dt is {r_val}")
    k_p = 0.333 
    k_i = 0.333
    k_d = 0
    k_t = 1
    dt_min = 1e-12
    dt_max = 5e-3 
    rho = 0.9
    #time_stores[2] = (  rho * ((r_val[0]/r_val[1])**k_p) * ((1/r_val[1])**k_i) * ((time_stores[1]/time_stores[0])**k_t) * time_stores[1]  )
    time_stores[2] = (  rho * ((1/r_val[1])**0.5) * time_stores[1]  )
    print (f"New timestep finded from rval is {time_stores[2]}")
    return max(min(time_stores[2],dt_max),dt_min)

def wlte(u_val,tau_abs,tau_rel,time_step_stores2):
    print (f"Timestep entering finding wlte is {time_step_stores2}")
    eta = 1 + (time_step_stores2[1]/time_step_stores2[2])
    #print (eta)
    sum = 0
    #print (u_val)
    for u00,u11,u22 in zip(u_val[0],u_val[1],u_val[2]):
        E1 = ((-1/eta)*u22) + ((1/(eta-1))*u11) - ((1/(eta*(eta-1)))*u00)
        sum += (E1/(tau_abs+(tau_rel*max(abs(u22),abs(u22+E1)))))**2 
    nnodes = len(u_val[0])
    r = ufl.sqrt((1/nnodes)*sum)
    #print (f"r_val finded is {r}")
    return r

#parameters
lmbda = 1.0e-02  # surface parameter
dt = 1.0e-09 # Current timestep
dt_2 = 1.0e-09 #Previous timestep
dt_1 = 1.0e-09 # Next timestep
epsilon = 1/64  # interface thickness
areainteng = 0.3 #Area specific interface energy
k = 0.05 #Dissipation term

#parameters for controller
time_step_store = np.array([dt_2,dt_1,dt])

#Parameter for r_val 
tau_abs = 1e-4
tau_rel = 1e-4

#Create array for storing u values
u_val = [None,None,None]
#Create array for storing r values
r_val = [None,None]
ii =0
#For plotting purpose
timesteps = [0]
E_stored = [0]
previous_energy = None  # Initialize previous_energy variable
storing_current_dt = [0]

#Creating mesh and function space
msh = create_unit_square(MPI.COMM_WORLD, 129, 129, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1)
ME = functionspace(msh, mixed_element([P1, P1]))
q, v = ufl.TestFunctions(ME)

u0 = Function(ME) # previous solution
u = Function(ME)  # current solution
temp_utot = Function(ME)

# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# Zero u
u.x.array[:] = 0.0
#Initial condition for concentration
#With pertubations
np.random.seed(30)
u.sub(0).interpolate(lambda x: 0.5 + 0.01 * (0.5 - np.random.rand(x.shape[1])))
u.x.scatter_forward()

dk = dolfinx.fem.Constant(msh, dt)

#Compute chemical potential 
c = ufl.variable(c)
f = areainteng*((6*(1/epsilon)*c**2*(1-c)**2))
dfdc = ufl.diff(f, c)

#Weak form equations
F0 = ((inner(c, q)- inner(c0, q))/dk) * dx + inner(grad(mu), grad(q)) * dx
F1 = inner(mu, v) * dx - inner(dfdc, v) * dx - 3*epsilon*areainteng*inner(grad(c), grad(v)) * dx
F = F0 + F1 

# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.atol = 1e-8  # Adjust the absolute tolerance
solver.rtol = 1e-5  # Adjust the relative tolerance
solver.error_on_nonconvergence = False
solver.max_it = 20

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}pc_type"] = "bjacobi"
opts[f"{option_prefix}absolute_tolerance"] = 1e-8
opts[f"{option_prefix}relative_tolerance"] = 1e-5
ksp.setFromOptions()
log.set_log_level(log.LogLevel.INFO)

# # Output file
file = XDMFFile(MPI.COMM_WORLD, "without_ncp_0.5.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 1e-8

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()
val = dolfinx.fem.Function(V0)
c = u.sub(0)
u0.x.array[:] = u.x.array #Stored value of n-1
bisection_count = 0

while (t < 0.2):
    print ("-----------------")
    dk.value = dt
    print (f"value of t is {t}")
    t += float(dk) #current time
    print(f"intial dt is {dt}")
    ii += 1
    (iter, converged) = solver.solve(u) #Call the function for the solver
    print (f"Time is : {t}" )
    if converged:
        temp_utot.x.array[:] = u0.x.array.copy()
        u0.x.array[:] = u.x.array
        #Storing three timestep values
        temp_u = u_val[0]
        u_val[0] = u_val[1]
        u_val[1] = u_val[2]
        u_val[2] = u.x.array[dofs].copy()
        print(f"Step: {ii} || Iterations: {iter}")
        
        if (ii<=10):
            energy = dolfinx.fem.form(f*dx)
            current_energy = assemble_scalar(energy)
            E_stored.append(current_energy)
            timesteps.append(t)
            storing_current_dt.append(dt)
            file.write_function(c, t)
            val.x.array[:] = u.x.array[dofs]
            val.x.scatter_forward()
            if ii == 10:
                r_val[0] = wlte(u_val.copy(),tau_abs,tau_rel,time_step_store.copy())
                print (r_val)
        #Activating the time controller 
        if ii>11:
            r_val[1] = wlte(u_val.copy(),tau_abs,tau_rel,time_step_store.copy()) 
            temp_t = time_step_store[0]
            dt_2 = time_step_store[1]
            dt_1 = time_step_store[2]
            New_timestep = adaptive_timestep(time_step_store.copy(),r_val.copy())
           
            if r_val[1] <= 1:
                storing_current_dt.append(dt)
                print ("Value of r is within tolerance")
                time_step_store = [dt_2, dt_1, New_timestep]
                print (f"Accepted timestep is {time_step_store}")
                r_val[0] = r_val[1]
                dt = time_step_store[2]
                dk.value = dt
                file.write_function(c, t)
                val.x.array[:] = u.x.array[dofs]
                val.x.scatter_forward()
                energy = dolfinx.fem.form(f*dx)
                current_energy = assemble_scalar(energy)
                E_stored.append(current_energy)
                timesteps.append(t)
                bisection_count = 0
                
                 
            elif r_val[1] > 1:
                print ("Value of r is not within tolerance")
                t -= float(dk)
                Nstep = dt_1*0.25
                print (f"new halved timestep is {Nstep}")
                #Reverse timestep to converged area
                #time_step_store[2] = New_timestep
                time_step_store = [temp_t, dt_2, Nstep]
                print (f"Restored timestep is {time_step_store}")
                #Reverse the field values
                u_val[2] = u_val[1]
                u_val[1] = u_val[0]
                u_val[0] = temp_u
                dt = time_step_store[2]
                dk.value = dt
                u0.x.array[:] = temp_utot.x.array
                u.x.array[:] = u0.x.array
    else:
        print ("Newton iteration not converged, reduce dt")   
        bisection_count += 0 
        if bisection_count > 5:
            print("Error: Too many bisections")
            break
        print("Error Halfing Time Step")
        t = t - float(dk)
        dt = dt / 2
        dk.value = dt
        print(f"New Time Step: {dt}")
        u.x.array[:] = u0.x.array
file.close()

#For saving values of concentration in file
file_path = './p_values_check.txt'
np.savetxt(file_path,val.x.array)

#Plotting energy vs timestep
import matplotlib.pyplot as plt
plt.plot(timesteps, E_stored, label='Energy vs. Time')
#plt.xscale('log')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy vs. Time')
plt.legend()
plt.show()

#Plotting energy vs timestep
import matplotlib.pyplot as plt
plt.plot(timesteps, storing_current_dt, label='Evolution of dt')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('simulation_Time')
plt.ylabel('delta_t')
plt.title('Delta_t vs. Sim_Time')
plt.legend()
plt.show()
