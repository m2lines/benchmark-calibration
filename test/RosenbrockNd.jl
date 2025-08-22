using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg

Random.seed!(1234)   # Fix random numbers globally

########## Set the inverse problem ##########
noise = 0.1
Ndim = 20
G(u) = vcat(u[2:end] .- u[1:end-1].^2, 1 .- u[1:end-1]) .+ noise * randn(2*Ndim-2)
true_u = ones(Ndim)
y = zeros(2*Ndim-2)
# Model of the observational error introduced above
Γ = (noise)^2*I
prior = constrained_gaussian("many_with_spread_1", 0, 1, -Inf, Inf, repeats=Ndim)

########### Initialize the EnsembleKalmanProcess ##########
N_ensemble = 30
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)

default_eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    verbose=false
)

vanilla_eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=false
)

########### Iterations with the ensemble ##########

N_iterations = 10
for i in 1:N_iterations
    params_default = get_ϕ_final(prior, default_eki)
    g_ens = hcat([G(params_default[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(default_eki, g_ens)

    params_vanilla = get_ϕ_final(prior, vanilla_eki)
    g_ens = hcat([G(params_vanilla[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(vanilla_eki, g_ens, deterministic_forward_map=false)

    solution_default = get_ϕ_mean_final(prior, default_eki)
    solution_vanilla = get_ϕ_mean_final(prior, vanilla_eki)

    println("Iteration $i. Default EKI error: ", norm(solution_default .- true_u), " Vanilla EKI error: ", norm(solution_vanilla .- true_u))
end