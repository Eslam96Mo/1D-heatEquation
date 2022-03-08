using CUDA, CairoMakie

α  = 0.01                                                                            # Diffusivity
L  = 0.1                                                                             # Length
M  = 66                                                                              # No.of steps
Δx = L/(M-1)                                                                         # x-grid spacing
Δt = Δx^2 / (2.0 * α * (Δx^2))                                         # Largest stable time step

function diffuse!(data, a, Δt, Δx)
    di  = view(data, 2:M-1)
    di1 = view(data, 1:M-2)
    di2 = view(data, 3:M  )
                                                # Stencil Computations
  
    @. di += α * Δt * (
        (di1 - 2 * di + di2)/Δx^2)                                               # Apply diffusion

    @. data[1, :] += α * Δt * (2*data[2, :] - 2*data[1, :])/Δx^2
    @. data[M, :] += α * Δt * (2*data[M-1, :] - 2*data[M, :])/Δx^2
                                          # update boundary condition (Neumann BCs)

end

domain     = zeros(M)                                                             # zeros Matrix 
domain_GPU = CuArray(convert(Array{Float32}, domain))                               # change the system to GPU instead of CPU
domain_GPU[32:33] .= 5                                                       # heat Source 

xspan = 0 : Δx : L
for i in 1:1000                                                                     # Apply the diffuse 1000 time to let the heat spread a long the rod       
    diffuse!(domain_GPU, α, Δt, Δx)
end

plot(xspan, [domain_GPU[:,1], domain_GPU[:,end]], label=["t=0" "t=Tf"], xlabel="Position x", ylabel="Temperature")