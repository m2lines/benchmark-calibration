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
Γ = Diagonal(fill((0.001)^2, 2))
prior = constrained_gaussian("two_with_spread_1", 0, 1, -Inf, Inf, repeats=2)

println("\n--- Problem setup ---")
println("True parameter: ", true_u)
println("Observations:   ", y)

########### Initialize the EnsembleKalmanProcess ##########
N_ensemble = 50
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)

eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, TransformInversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true,
)

vanilla_eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true
)

######## Get ensemble of parameters ########

params_i = get_u_final(eki)
g_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

println("\n--- Forward model evaluation ---")
println("Shape of g_ens (outputs): ", size(g_ens))

############ Verify method against hardcoded baseline ###############
# anomalies
g_prime = g_ens .- mean(g_ens; dims=2)                      # (d × N)
params_prime = params_i .- mean(params_i; dims=2)           # (p × N)

# Cross-covariance between parameters and outputs
C_pg = (params_prime * g_prime') / N_ensemble               # (p × d)

# Auto-covariance of outputs
C_gg = (g_prime * g_prime') / N_ensemble + 1e-6I            # (d × d)

# Kalman gain (p × d)
K = C_pg * inv(C_gg)

# Update (broadcast matmul across ensembles)
params_new = params_i .- K * g_ens                          # (p × N)

println("\n--- Baseline Kalman update ---")
println("Shape of Kalman gain K: ", size(K))

############# Predict using EnsembleKalmanProcesses #################
Random.seed!(1234)   # Fix random numbers globally
update_ensemble!(eki, g_ens, deterministic_forward_map=true)
params_etki = get_u_final(eki)

Random.seed!(1234)   # Fix random numbers globally
update_ensemble!(vanilla_eki, g_ens, deterministic_forward_map=true)
params_vanilla = get_u_final(vanilla_eki)

############## Compare the two methods #################
rel_err_etki = norm(params_etki - params_new) / norm(params_i)
rel_err_vanilla = norm(params_vanilla - params_new) / norm(params_i)
rel_err_etki_vanilla = norm(params_vanilla - params_etki) / norm(params_i)

println("\n===== Comparison =====")

Pkg.status("EnsembleKalmanProcesses")

println("Relative error ETKI - deterministic: ", rel_err_etki, " Expected output (v.2.5.0): ", 0.0003281747006106274)
println("Relative error vanilla EKI - deterministic: ", rel_err_vanilla, " Expected output (v.2.5.0): ", 0.0003382036240117189)
println("Relative error vanilla EKI - ETKI: ", rel_err_etki_vanilla, " Expected output (v.2.5.0): ", 0.0004610289494884395)
println("==============================================")