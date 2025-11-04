using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg

Random.seed!(1234)   # Fix random numbers globally

println("===== Ensemble Kalman Process Test =====")

########## Set the inverse problem ##########
G(u) = (exp.(-u) .- u .+ 0.001 * randn(length(u)))
y = [0,0]
true_u = [0.56714329,0.56714329]
# Model of the observational error introduced above
Γ = (0.001)^2*I
prior = constrained_gaussian("two_with_spread_1", 0, 1, -Inf, Inf, repeats=2)

println("\n--- Problem setup ---")
println("True parameter: ", true_u)
println("Observations:   ", y)

########### Initialize the EnsembleKalmanProcess ##########
N_ensemble = 50
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)

# Only data-misfit
control_eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    verbose=true,
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
)

# Data misfit + Nesteros
test_eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true
)

######## Get ensemble of parameters ########

params_i = get_u_final(control_eki)
g_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

println("\n--- Forward model evaluation ---")
println("Shape of g_ens (outputs): ", size(g_ens))


update_ensemble!(control_eki, g_ens, deterministic_forward_map=false)
params_control = get_u_final(control_eki)

update_ensemble!(test_eki, g_ens, deterministic_forward_map=false)
params_test = get_u_final(test_eki)

############## Compare the two methods #################
rel_err = norm(params_control - params_test) / norm(params_i)

println("\n===== Comparison =====")

Pkg.status("EnsembleKalmanProcesses")

println("Relative error between Nesterov and Nesterov+DMC: ", rel_err, " Expected output (v.2.5.0): ", 0.0)
println("==============================================")
