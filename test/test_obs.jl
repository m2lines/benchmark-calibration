using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using Pkg

Random.seed!(1234)   # Fix random numbers globally

println("===== Ensemble Kalman Process Test =====")

########## Set the inverse problem ##########
G(u) = (exp.(-u) .- u)
y = [1,1]
true_u = [0.56714329,0.56714329]
# Model of the observational error introduced above
Γ = (0.001)^2*I
prior = constrained_gaussian("two_with_spread_1", 0, 1, -Inf, Inf, repeats=2)

println("\n--- Problem setup ---")
println("True parameter: ", true_u)
println("Observations:   ", y)

########### Initialize the EnsembleKalmanProcess ##########
N_ensemble = 5
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)

eki_obs = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    verbose=false
)

eki_zero = EnsembleKalmanProcess(
    initial_ensemble, [0,0], Γ, Inversion(),
    verbose=false
)

######## Get ensemble of parameters ########
N_iterations = 10
for k in 1:N_iterations
    params_i = get_u_final(eki_obs)
    g_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)
    update_ensemble!(eki_obs, g_ens, deterministic_forward_map=false)

    params_i = get_u_final(eki_zero)
    g_ens = hcat([G(params_i[:, i]) .- y for i in 1:N_ensemble]...)
    update_ensemble!(eki_zero, g_ens, deterministic_forward_map=false)
end

############## Compare the two methods #################
params_obs = get_u_final(eki_obs)
params_zero = get_u_final(eki_zero)
rel_err = norm(params_zero - params_obs) / norm(params_zero)

println("\n===== Comparison =====")

# show package status
Pkg.status("EnsembleKalmanProcesses")

# then print the relative error
println("Relative error between two obs models: ", rel_err)
println("Expected output (v.2.5.0): 1.021872746011873e-14")