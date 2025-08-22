using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg
using Plots

Random.seed!(1234)   # Fix random numbers globally

########## Set the inverse problem ##########
a = sqrt(7/5.)
b = 1.
noise = 0.5
G(u) = [a - u[1], b * (u[2] - u[1]^2)] .+ noise * randn(2)
true_u = [a,a^2]
y = [0,0]
# Model of the observational error introduced above
Γ = (noise)^2*I
prior = constrained_gaussian("two_with_spread_1", 0, 5, -Inf, Inf, repeats=2)

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

    plt = scatter(params_default[1, :], params_default[2, :],
        xlabel = "Parameter 1",
        ylabel = "Parameter 2",
        label = "Default EKI",
        xlims = (-6, 6),
        ylims = (-6, 6),
        legend = true)
    scatter!(params_vanilla[1, :], params_vanilla[2, :],
        label = "Vanilla EKI")
    scatter!(plt, [true_u[1]], [true_u[2]], label="True solution", color=:red, markersize=8)
    display(plt)  # show each figure

    savefig(plt, "Rosenbrock2d/iteration_$i.png")
end