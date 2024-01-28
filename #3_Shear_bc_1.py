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
msh=mesh.create_unit_square(comm=MPI.COMM_WORLD,nx=1,ny=1,cell_type=mesh.CellType.quadrilateral)

#Creating function space
element = ufl.VectorElement("Lagrange", msh.ufl_cell(),1)
V=fem.FunctionSpace(msh, element)   

#Applying bc
def bottom(x):
    return np.isclose(x[1],0)
bottom_value = np.array((0,)*msh.geometry.dim, dtype=default_scalar_type)
bottom_dof = fem.locate_dofs_geometrical(V,bottom)
bc_bottom = fem.dirichletbc(bottom_value,bottom_dof,V)

def top(x):
    print (np.isclose(x[1],1))
    return np.isclose(x[1],1)
top_facets = mesh.locate_entities_boundary(msh,msh.topology.dim-1,top)
top_dof = fem.locate_dofs_topological(V.sub(0),msh.topology.dim-1,top_facets)
bc_top = fem.dirichletbc(default_scalar_type(0.5),top_dof,V.sub(0))

def left(x):
    return np.isclose(x[0],0)

left_facets = mesh.locate_entities_boundary(msh,msh.topology.dim-1,left)
left_dof = fem.locate_dofs_topological(V.sub(1),msh.topology.dim-1,left_facets)
bc_left = fem.dirichletbc(default_scalar_type(0),left_dof,V.sub(1))

def right(x):
    return np.isclose(x[0],1)

right_facets = mesh.locate_entities_boundary(msh,msh.topology.dim-1,right)
right_dof = fem.locate_dofs_topological(V.sub(1),msh.topology.dim-1,right_facets)
bc_right = fem.dirichletbc(default_scalar_type(0),right_dof,V.sub(1))

bcs = [bc_bottom,bc_top,bc_left,bc_right]

#Defining traction and body forces
T=fem.Constant(msh,default_scalar_type((0,0)))
B=fem.Constant(msh,default_scalar_type((0,0)))

#Defining test and solution function
v=ufl.TestFunction(V)
u=fem.Function(V)
print (u.x.array)
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
#Defining first piola stress
#P = 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2

# Hyper-elasticity
P = ufl.diff(psi, F)

ds = ufl.Measure('ds')
dx = ufl.Measure('dx')

#Defining the weak form
E = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds

#Solving Nonlinear problem
problem=NonlinearProblem(E,u,bcs)

#Solving using newton solver
solver=NewtonSolver(msh.comm,problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

solver.solve(u)

a=u.x.array
print(a)   
''''
with io.XDMFFile(msh.comm, "shear_deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    u.name = "Deformation"
    xdmf.write_function(u)

'''''
element2 = ufl.TensorElement("DG",msh.ufl_cell(),0)
V2=fem.FunctionSpace(msh,element2)
s = ufl.TrialFunction(V2)
t = ufl.TestFunction(V2)

a = ufl.inner(s, t)*dx
L = ufl.inner(P, t)*dx

problem2 = LinearProblem(a,L)
stress = problem2.solve()

s_val=stress.x.array
print (s_val)