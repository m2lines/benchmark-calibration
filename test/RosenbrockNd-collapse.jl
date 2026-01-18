using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg

#Random.seed!(123)   # Fix random numbers globally

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

unscented_utki = EnsembleKalmanProcess(
    y, Γ,
    TransformUnscented(prior, impose_prior=true),
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

vanilla_etki = EnsembleKalmanProcess(
    initial_ensemble,
    y, Γ,
    TransformInversion(),
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

dmc_etki = EnsembleKalmanProcess(
    initial_ensemble,
    y, Γ,
    TransformInversion(),
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

params_my = initial_ensemble

########### Iterations with the ensemble ##########
using Printf

N_iterations = 5
for i in 1:N_iterations
    global params_my

    # My method of regularization
    params = params_my
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)

    g_prime = g_ens .- mean(g_ens; dims=2)                  # (d × N)
    params_prime = params .- mean(params; dims=2)           # (p × N)

    # Cross-covariance between parameters and outputs
    C_pg = (params_prime * g_prime') / N_ensemble           # (p × d)

    # Auto-covariance of outputs
    C_gg = (g_prime * g_prime') / N_ensemble                # (d × d)

    trace_Cgg = tr(C_gg)
    trace_Gamma = noise^2 * Ndim
    alpha = trace_Cgg / trace_Gamma

    # Kalman gain (p × d)
    K = C_pg * inv(C_gg + alpha * Γ)

    # Update ensemble (Note: obs vector is zero)
    params_my = params_my .+ K * (0 .- g_ens)               # (p × N)

    # Baselines
    params = get_ϕ_final(prior, unscented_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(unscented_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, unscented_utki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(unscented_utki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, unscented_dmc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(unscented_dmc_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, vanilla_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(vanilla_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, vanilla_etki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(vanilla_etki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, dmc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(dmc_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, dmc_etki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(dmc_etki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, loc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(loc_eki, g_ens, deterministic_forward_map=false)

    params = get_ϕ_final(prior, loc_dmc_eki)
    g_ens = hcat([G(params[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(loc_dmc_eki, g_ens, deterministic_forward_map=false)

    my_solution = mean(params_my; dims=2)
    unscented_solution = get_ϕ_mean_final(prior, unscented_eki)
    unscented_utki_solution = get_ϕ_mean_final(prior, unscented_utki)
    unscented_dmc_solution = get_ϕ_mean_final(prior, unscented_dmc_eki)
    vanilla_solution = get_ϕ_mean_final(prior, vanilla_eki)
    vanilla_etki_solution = get_ϕ_mean_final(prior, vanilla_etki)
    dmc_solution = get_ϕ_mean_final(prior, dmc_eki)
    dmc_etki_solution = get_ϕ_mean_final(prior, dmc_etki)
    loc_solution = get_ϕ_mean_final(prior, loc_eki)
    loc_dmc_solution = get_ϕ_mean_final(prior, loc_dmc_eki)
    println("Errors at iteration $i")
    println("===================================")

    @printf("UKI:         %.3f\n", norm(unscented_solution .- true_u))
    @printf("UTKI:         %.3f\n", norm(unscented_utki_solution .- true_u))
    @printf("EKI+LOC:     %.3f\n", norm(loc_solution .- true_u))

    println("")
    @printf("EKI:         %.3f\n", norm(vanilla_solution .- true_u))
    @printf("ETKI:         %.3f\n", norm(vanilla_etki_solution .- true_u))

    println("")

    @printf("EKI+DMC:     %.3f\n", norm(dmc_solution .- true_u))
    @printf("ETKI+DMC:     %.3f\n", norm(dmc_etki_solution .- true_u))
    @printf("UKI+DMC:     %.3f\n", norm(unscented_dmc_solution .- true_u))
    @printf("EKI my:      %.3f\n", norm(my_solution .- true_u))

    println("")
    @printf("EKI+LOC+DMC: %.3f\n", norm(loc_dmc_solution .- true_u))
    
    println("===================================")
    println("")
end
