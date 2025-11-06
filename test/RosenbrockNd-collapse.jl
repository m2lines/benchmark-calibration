using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg

Random.seed!(1234)   # Fix random numbers globally

########## Set the inverse problem ##########
noise = 0.01
Ndim = 13
G(u) = vcat(u[2:end] .- u[1:end-1].^2, 1 .- u[1:end-1]) .+ noise * randn(2*Ndim-2)
true_u = ones(Ndim)
y = zeros(2*Ndim-2)
# Model of the observational error introduced above
Γ = (noise)^2*I
prior = constrained_gaussian("many_with_spread_1", 0, 1, -Inf, Inf, repeats=Ndim)

########### Initialize the EnsembleKalmanProcess ##########
N_ensemble = 27
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)

unscented_eki = EnsembleKalmanProcess(
    y, Γ,
    Unscented(prior, impose_prior=true),
    scheduler = DefaultScheduler(1),
    verbose=false
)

unscented_dmc_eki = EnsembleKalmanProcess(
    y, Γ,
    Unscented(prior, impose_prior=true),
    verbose=false
)

vanilla_eki = EnsembleKalmanProcess(
    initial_ensemble,
    y, Γ,
    Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=false
)

dmc_eki = EnsembleKalmanProcess(
    initial_ensemble,
    y, Γ,
    Inversion(),
    #scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=false
)

loc_eki = EnsembleKalmanProcess(
    initial_ensemble,
    y, Γ,
    Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    #localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=false
)

loc_dmc_eki = EnsembleKalmanProcess(
    initial_ensemble,
    y, Γ,
    Inversion(),
    #scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    #localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=false
)

########### Iterations with the ensemble ##########
using Printf

N_iterations = 5
for i in 1:N_iterations
    params = get_ϕ_final(prior, unscented_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(unscented_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, unscented_dmc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(unscented_dmc_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, vanilla_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(vanilla_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, dmc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(dmc_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, loc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(loc_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, loc_dmc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(loc_dmc_eki, g_ens, deterministic_forward_map=false)

    unscented_solution = get_ϕ_mean_final(prior, unscented_eki)
    unscented_dmc_solution = get_ϕ_mean_final(prior, unscented_dmc_eki)
    vanilla_solution = get_ϕ_mean_final(prior, vanilla_eki)
    dmc_solution = get_ϕ_mean_final(prior, dmc_eki)
    loc_solution = get_ϕ_mean_final(prior, loc_eki)
    loc_dmc_solution = get_ϕ_mean_final(prior, loc_dmc_eki)
    println("Errors at iteration $i")
    println("===================================")

    @printf("UKI:         %.3f\n", norm(unscented_solution .- true_u))
    @printf("EKI+LOC:     %.3f\n", norm(loc_solution .- true_u))
    @printf("UKI+DMC:     %.3f\n", norm(unscented_dmc_solution .- true_u))
    @printf("EKI+LOC+DMC: %.3f\n", norm(loc_dmc_solution .- true_u))

    println("")
    @printf("EKI:         %.3f\n", norm(vanilla_solution .- true_u))
    @printf("EKI+DMC:     %.3f\n", norm(dmc_solution .- true_u))
    
    println("===================================")
    println("")
end
