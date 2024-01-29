#Importing libraries
import numpy as np
import dolfinx  
import ufl
from dolfinx import fem, io, mesh, plot,default_scalar_type
from ufl import ds, dx, grad, inner
from dolfinx.fem.petsc import NonlinearProblem,LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc

#Creating mesh
msh = mesh.create_rectangle (comm=MPI.COMM_WORLD, points=((0.0, 0.0), (6.0,1.0)), n=(384,64), cell_type=mesh.CellType.quadrilateral)

#Creating function space
element = ufl.VectorElement("Lagrange", msh.ufl_cell(),1)
V=fem.FunctionSpace(msh, element)

#Applying boundary condition to beam
def left_corner(x):
    return np.logical_and(np.isclose(x[1],0.0),np.isclose(x[0],0.0))
#u_zero = np.array((0,) * msh.geometry.dim, dtype=default_scalar_type)
#bc = fem.dirichletbc(u_zero, fem.locate_dofs_geometrical(V, left_corner), V)
u_zero = np.array((0,) * 2, dtype=default_scalar_type)
left_corner_vertex = mesh.locate_entities_boundary(msh,0,left_corner)
left_dof = fem.locate_dofs_topological(V,0,left_corner_vertex)
bc_left_corner = fem.dirichletbc(u_zero,left_dof,V)

def right_corner(x):
    return np.logical_and(np.isclose(x[1],0.0),np.isclose(x[0],6.0))
right_corner_vertex = mesh.locate_entities_boundary(msh,0,right_corner)
right_dof =  fem.locate_dofs_topological(V.sub(1),msh.topology.dim-2,right_corner_vertex)
bc_right_corner = fem.dirichletbc(default_scalar_type(0),right_dof,V.sub(1))

bcs = [bc_left_corner, bc_right_corner]

def top(x):
    return np.logical_and(np.isclose(x[1], 1), np.logical_and(x[0] >= 2.96875, x[0] <= 3.03125)) 
top_facets = mesh.locate_entities_boundary(msh,msh.topology.dim-1,top)
marked_facets = top_facets
marked_values = np.full_like(top_facets,1)
facet_tag = mesh.meshtags(msh,msh.topology.dim-1,marked_facets, marked_values)

#Defining traction and body forces
T=fem.Constant(msh,default_scalar_type((0,-1000)))
B=fem.Constant(msh,default_scalar_type((0,0)))

#Defining test and solution function
v=ufl.TestFunction(V)
u=fem.Function(V)

#Defining kinematic quantities
#Spatial dimension
d=len(u)
#Identity tensor
I=ufl.variable(ufl.Identity(d))
#Deformation gradient
F=ufl.variable(I+grad(u))
# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)
# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# Elasticity parameters
E =default_scalar_type(1e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(msh, E / (2 * (1 + nu)))
lmbda = fem.Constant(msh, E * nu / ((1 + nu) * (1 - 2 * nu)))

psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Stress
# Hyper-elasticity
P = ufl.diff(psi, F)

ds = ufl.Measure('ds', domain=msh,subdomain_data=facet_tag)
dx = ufl.Measure('dx')

#Defining the weak form
A = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(1)

#Solving Nonlinear problem
problem=NonlinearProblem(A,u,bcs)

#Solving using newton solver
solver=NewtonSolver(msh.comm,problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

solver.solve(u)
with io.XDMFFile(msh.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    u.name = "Deformation"
    xdmf.write_function(u)