using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg

Random.seed!(1234)   # Fix random numbers globally

println("===== Ensemble Kalman Process Test =====")

########## Set the baseline inverse problem ##########
G_base(u) = [u[1], u[2]]
y_base = [0, 0]
Gamma_base = (0.01)^2 * I

########## Set the inflated inverse problem ##########
G(u) = [u[1], u[2], u[2]]
y = [0, 0, 0]
Gamma = Diagonal([0.01^2, (0.01*sqrt(2))^2, (0.01*sqrt(2))^2])

# Set the initial ensemble
N_ensemble = 50
prior = constrained_gaussian("two_with_spread_1", 0, 1, -Inf, Inf, repeats=2)
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)

base_eki = EnsembleKalmanProcess(
    initial_ensemble, y_base, Gamma_base, Inversion(),
    scheduler = DefaultScheduler(1),
    #accelerator = DefaultAccelerator(),
    #localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true
)

eki = EnsembleKalmanProcess(
    initial_ensemble, y, Gamma, Inversion(),
    scheduler = DefaultScheduler(1),
    #accelerator = DefaultAccelerator(),
    #localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true
)

######## Iterate predictions ########
N_iterations = 2
for j in 1:N_iterations
    params_base = get_u_final(base_eki)
    g_base = hcat([G_base(params_base[:, i]) for i in 1:N_ensemble]...)

    params = get_u_final(eki)
    g = hcat([G(params[:, i]) for i in 1:N_ensemble]...)

    update_ensemble!(base_eki, g_base, deterministic_forward_map=false)
    update_ensemble!(eki, g, deterministic_forward_map=false)
end

params_base = get_u_final(base_eki)
params = get_u_final(eki)

############## Compare the two methods #################
rel_err = norm(params_base - params) / norm(params_base)

println("\n===== Comparison =====")

Pkg.status("EnsembleKalmanProcesses")

println("Relative error: ", rel_err, " Expected output (v.2.5.0): ", 9.238865625065e-13)
println("==============================================")