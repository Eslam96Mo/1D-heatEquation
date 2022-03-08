using LinearAlgebra, Plots, SparseArrays, DifferentialEquations

λ = 45.0          # Conductivity
ρ = 7800.0;       # Density
cap = 480.0;      # Specific heat capacitivity
 α = λ/(cap * ρ)   # Diffusivity
 L = 0.1           # Length of rod


#Boundary conditions
h = 10            # Conduction coefficient
ϵ = 0.6;          # Emissivity
sb = 5.67*10^(-8) # Stefan-Boltzmann constant
 k = ϵ * sb        # Radiation coefficient

# Discretization  
dx = L/100;                     # Spatial sampling: o--o--o--o--o--o
N = round(Int, (L / dx + 1));   # Number of elements


 Tf = 300.0    # Final time
 dt = 10^(-2); # Time step width
 t_len = round(Int, (Tf/dt + 1))


#Laplace operator
 M = spdiagm(-1 => ones(N-1), 0 => -2*ones(N), 1 => ones(N-1))
M[1,2] = 2
M[end,end-1] = 2

 Mx = zeros(N)
 C = spzeros(N,2) # Boundary Conditions: Output / Outflow

C[1,1] = 1	  # Left side
C[end,2] = 1  # Right side

q = α*dt/(dx^2);
if q > 0.5
  error("Numerical stability is NOT guaranteed! STOP program!")
end

# Temperatures
 ϑ₀   = 400.0; # Initial temperature
 ϑamb = 273.0; # Ambient temperature

# Define initial temperature
function init_temperature(ϑoffset, scale, dx, num_points)
	a = scale
	b = ϑoffset
	
	L = (num_points-1)*dx
	x = 0 : dx : L
	
	ϑinit = a*sin.(x*2*pi/L) .+ b
	
	return ϑinit
end


# Define the discretized PDE as an ODE function
function heat_eq(dθ,θ,p,t)
 
  λ = p[1]
  ρ = p[2]   
  cap = p[3]
 
  mul!(Mx,M,θ[1:end]) # means: Mx = M * Θ[1:end]

  # Now: zero Neumann boundary conditions
  bc = zeros(2)

  # Later: Natural boundary condition: heat conduction + radiation
  bc[1] =  -1*(h*(θ[1] - ϑamb) + k*(θ[1]^4 - ϑamb^4)); 
  bc[2] =  -1*(h*(θ[end] - ϑamb) + k*(θ[end]^4 - ϑamb^4)); 	
  
  flux_out =(2*dx/λ)*C*bc                     # Temperature input and output
  
  dθ .= λ/(ρ*cap) * 1/(dx^2) * (Mx + flux_out)   # Integration of temperatures

end


θinit = init_temperature(ϑ₀, 30.0, dx, N)

tspan = (0.0, Tf) # Time span
param = [λ, ρ, cap] # Parameter that shall be learned

prob = ODEProblem( heat_eq, θinit,tspan, param )
sol = solve(prob,Euler(),dt=dt,progress=true,save_everystep=true,save_start=true) # use 'save_everystep' only for small systems with few timesteps 

xspan = 0 : dx : L

plot(xspan, [sol[:,1], sol[:,end]], label=["t=0" "t=Tf"], xlabel="Position x", ylabel="Temperature")

plot(sol.t,[sol[1,:], sol[end,:]], label=["Left" "Right"], xlabel="Time t", ylabel="Temperature", legend=:bottomright)

heatmap(sol.t, xspan, sol[:,:], xlabel="Time t", ylabel="Position x")