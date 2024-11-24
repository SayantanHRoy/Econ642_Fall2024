using Plots, QuantEcon
include("SGU.jl")
using .SGU
using Parameters, SymEngine,Statistics

function IRBC_model(; pi_temp=0.5, rho_h_temp=0.9, rho_f_temp=0.9, rho_til_temp=0.0, stdeps_h_temp=0.01, stdeps_f_temp=0.01)
    # Step 1: Declare parameters and variables
    params_keys = "beta, delta, gamma, alpha, kappa_h, kappa_f, psi, phi_k, phi_b, pi, rho_h, rho_f, rho_til, stdeps_h, stdeps_f"
    vars_keys = "C_h, C_f, K_h, K_f, K_fu_h, K_fu_f, Y_h, Y_f, H_h, H_f, lambda_h, lambda_f, Z_h, Z_f, B_h, B_f, PB, IV_h, IV_f"
    
    params, vars, varsp = SGU.create_variables(params_keys, vars_keys)

    # Unpack for symbolic manipulation
    @unpack beta, delta, gamma, alpha, kappa_h, kappa_f, psi, phi_k, phi_b, pi, rho_h, rho_f, rho_til, stdeps_h, stdeps_f = params
    @unpack C_h, C_f, K_h, K_f, K_fu_h, K_fu_f, Y_h, Y_f, H_h, H_f, lambda_h, lambda_f, Z_h, Z_f, B_h, B_f, PB, IV_h, IV_f = vars
    @unpack C_hp, C_fp, K_hp, K_fp, K_fu_hp, K_fu_fp, Y_hp, Y_fp, H_hp, H_fp, lambda_hp, lambda_fp, Z_hp, Z_fp, B_hp, B_fp, PBp, IV_hp, IV_fp = varsp

    shocks = "Z_h,Z_f"
    std_eps = "stdeps_h,stdeps_f"

    # Step 2: Model equations
    F = Array{Basic}(undef, 19)
    F .= 0

    # Home country equations
    F[1] = exp(C_h)^(-gamma) - exp(lambda_h)
    
    F[2] = kappa_h*exp(H_h)^(1/psi) - 
           exp(lambda_h)*exp(Z_h)*(1-alpha)*(exp(K_h)/exp(H_h))^alpha
    
    F[3] = exp(lambda_h)*(1 + phi_k*(exp(K_hp) - exp(K_h))) - 
           beta*exp(lambda_hp)*(1-delta + 
           exp(Z_hp)*alpha*(exp(K_hp)/exp(H_hp))^(alpha-1) + 
           phi_k*(exp(K_fu_hp) - exp(K_hp)))
    
    F[4] = beta*exp(lambda_hp) - 
           exp(lambda_h)*(exp(PB) + phi_b*exp(PB)*B_hp)
    
    F[5] = exp(C_h) + exp(K_hp) + exp(PB)*B_hp - 
           (1-delta)*exp(K_h) - 
           exp(Z_h)*exp(K_h)^alpha*exp(H_h)^(1-alpha) - 
           B_h + (phi_k/2)*(exp(K_hp) - exp(K_h))^2 + 
           (phi_b/2)*exp(PB)*(B_hp)^2
    
    F[6] = exp(Y_h) - exp(Z_h)*exp(K_h)^alpha*exp(H_h)^(1-alpha)
    
    F[7] = exp(IV_h) - exp(K_hp) + (1-delta)*exp(K_h)
    
    F[8] = exp(K_fu_h) - exp(K_hp)

    # Foreign country equations
    F[9] = exp(C_f)^(-gamma) - exp(lambda_f)
    
    F[10] = kappa_f*exp(H_f)^(1/psi) - 
            exp(lambda_f)*exp(Z_f)*(1-alpha)*(exp(K_f)/exp(H_f))^alpha
    
    F[11] = exp(lambda_f)*(1 + phi_k*(exp(K_fp) - exp(K_f))) - 
            beta*exp(lambda_fp)*(1-delta + 
            exp(Z_fp)*alpha*(exp(K_fp)/exp(H_fp))^(alpha-1) + 
            phi_k*(exp(K_fu_fp) - exp(K_fp)))
    
    F[12] = beta*exp(lambda_fp) - 
            exp(lambda_f)*(exp(PB) + phi_b*exp(PB)*B_fp)
    
    F[13] = exp(C_f) + exp(K_fp) + exp(PB)*B_fp - 
            (1-delta)*exp(K_f) - 
            exp(Z_f)*exp(K_f)^alpha*exp(H_f)^(1-alpha) - 
            B_f + (phi_k/2)*(exp(K_fp) - exp(K_f))^2 + 
            (phi_b/2)*exp(PB)*(B_fp)^2
    
    F[14] = exp(Y_f) - exp(Z_f)*exp(K_f)^alpha*exp(H_f)^(1-alpha)
    
    F[15] = exp(IV_f) - exp(K_fp) + (1-delta)*exp(K_f)
    
    F[16] = exp(K_fu_f) - exp(K_fp)

    # Bond market clearing
    F[17] = pi*B_h + (1-pi)*B_f

    # Technology processes
    F[18] = Z_hp - rho_h*Z_h - rho_til*Z_f
    F[19] = Z_fp - rho_f*Z_f - rho_til*Z_h

    # Define state variables and controls
    statevar = [K_h; K_f; B_h; Z_h; Z_f]
    controlvar = [C_h; C_f; H_h; H_f; Y_h; Y_f; IV_h; IV_f; 
                 lambda_h; lambda_f; B_f; PB; K_fu_h; K_fu_f]

    # Linearize the model
    FF = SGU.linearize(F, statevar, controlvar, params, vars, shocks, std_eps)

    # Step 3: Assign parameter values
    alpha = 0.33
    delta = 0.025
    beta = 0.98
    gamma = 2.0
    psi = 1
    phi_k = 0.1
    phi_b = 0.0001
    pi = pi_temp
    rho_h = rho_h_temp
    rho_f = rho_f_temp
    rho_til = rho_til_temp
    stdeps_h = stdeps_h_temp
    stdeps_f = stdeps_f_temp

    # Step 4: Compute steady state
    K_ss = ((1/beta - (1-delta))/alpha)^(1/(alpha-1)) * (1/3)
    H_ss = 1/3
    Y_ss = K_ss^alpha * H_ss^(1-alpha)
    IV_ss = delta*K_ss
    C_ss = Y_ss - IV_ss
    lambda_ss = C_ss^(-gamma)
    B_ss = 0.0
    PB_ss = beta

    # use h_ss to define kappa 
    kappa = lambda_ss*(1-alpha)*K_ss^alpha*H_ss^(-alpha)*H_ss^(-1/psi)
    kappa_h = kappa
    kappa_f = kappa

    # Pack parameters
    @pack! params = alpha, delta, gamma, beta, psi, phi_k, phi_b, pi, rho_h, rho_f, rho_til, stdeps_h, stdeps_f, kappa_h, kappa_f

    # Assign steady state values (in same order as vars_keys)
    C_h = log(C_ss)
    C_f = log(C_ss)
    K_h = log(K_ss)
    K_f = log(K_ss)
    K_fu_h = log(K_ss)
    K_fu_f = log(K_ss)
    Y_h = log(Y_ss)
    Y_f = log(Y_ss)
    H_h = log(H_ss)
    H_f = log(H_ss)
    lambda_h = log(lambda_ss)
    lambda_f = log(lambda_ss)
    Z_h = 0.0
    Z_f = 0.0
    B_h = B_ss
    B_f = B_ss
    PB = log(PB_ss)
    IV_h = log(IV_ss)
    IV_f = log(IV_ss)

    # Pack variables 
    @pack! vars = C_h, C_f, K_h, K_f, K_fu_h, K_fu_f, Y_h, Y_f, H_h, H_f, 
                  lambda_h, lambda_f, Z_h, Z_f, B_h, B_f, PB, IV_h, IV_f

    @pack! FF = vars, params

    # Step 5: Solve model
    G = SGU.num_eval(FF, disp=true)
    return G
end

# Solve model
G1 = IRBC_model(pi_temp = 0.7,rho_h_temp = 0.99999,rho_f_temp = 0.99999) 
G2 = IRBC_model(pi_temp = 0.7,rho_h_temp = 0.9,rho_f_temp = 0.9) 


# Report linear decision rules
println("\nState transition matrix (hx) for G1:")
display(G1.hx)

# Report linear decision rules
println("\nState transition matrix (hx) for G2:")
display(G2.hx)



# Set up IRF calculations
T = 40  # periods
x0 = zeros(5)  # 5 state variables
x0[G1.ix[:Z_h]] = 1.0  # unit shock to home productivity

# Generate IRFs
_, IRy, IRx = SGU.ir(G1.gx, G1.hx, x0, T)

# Create 2x2 panel of plots
p1 = plot(IRy[:,G1.iy[:Y_h]], label="Home", title="Output", linewidth=2)
plot!(p1, IRy[:,G1.iy[:Y_f]], label="Foreign", linestyle=:dash)

p2 = plot(IRy[:,G1.iy[:C_h]], label="Home", title="Consumption", linewidth=2)
plot!(p2, IRy[:,G1.iy[:C_f]], label="Foreign", linestyle=:dash)

p3 = plot(IRy[:,G1.iy[:IV_h]], label="Home", title="Investment", linewidth=2)
plot!(p3, IRy[:,G1.iy[:IV_f]], label="Foreign", linestyle=:dash)

p4 = plot(IRy[:,G1.iy[:H_h]], label="Home", title="Hours", linewidth=2)
plot!(p4, IRy[:,G1.iy[:H_f]], label="Foreign", linestyle=:dash)

# Combine plots
plot(p1, p2, p3, p4, layout=(2,2), size=(800,600))




# Simulate the model
Xsim_g1, Ysim_g1 = SGU.simu_1st(G1.gx, G1.hx, G1.nETASHOCK, 1500)
Xsim_g2, Ysim_g2 = SGU.simu_1st(G2.gx, G2.hx, G2.nETASHOCK, 1500)

Tdrop = 500

# Function to HP filter and compute moments
function compute_moments(Ysim, G)
    # Get the data after dropping initial periods
    Y_h = hp_filter(Ysim[Tdrop+1:end, G.iy[:Y_h]], 1600)[2]
    Y_f = hp_filter(Ysim[Tdrop+1:end, G.iy[:Y_f]], 1600)[2]
    C_h = hp_filter(Ysim[Tdrop+1:end, G.iy[:C_h]], 1600)[2]
    C_f = hp_filter(Ysim[Tdrop+1:end, G.iy[:C_f]], 1600)[2]
    IV_h = hp_filter(Ysim[Tdrop+1:end, G.iy[:IV_h]], 1600)[2]
    IV_f = hp_filter(Ysim[Tdrop+1:end, G.iy[:IV_f]], 1600)[2]

    # Standard deviations
    println("\nStandard Deviations:")
    println("Home Output: ", std(Y_h))
    println("Foreign Output: ", std(Y_f))
    println("Home Consumption: ", std(C_h))
    println("Foreign Consumption: ", std(C_f))
    println("Home Investment: ", std(IV_h))
    println("Foreign Investment: ", std(IV_f))

    # Cross-country correlations
    println("\nCross-country Correlations:")
    println("Output: ", cor(Y_h, Y_f))
    println("Consumption: ", cor(C_h, C_f))
    println("Investment: ", cor(IV_h, IV_f))
end


compute_moments(Ysim_g1, G1)
compute_moments(Ysim_g2, G2)