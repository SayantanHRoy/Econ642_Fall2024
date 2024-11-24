#Implementation of log-linearization of DSGE models using SymEngine
#Original code in MATLAB from Schmitt-Grohe and Uribe's package
#Author: Hanjo Terry Kim (terryhanjokim@gmail.com) 
#This is translated to Julia with some syntax changes to make the implementation more Julia 
#All comments and feedback can be sent to terryhanjokim@gmail.com 
#For now, it only includes first-order perturbation, but I may include other features in the future 
#Last Updated: 9/3/2024 

module SGU 

using SymEngine, Parameters, Random, LinearAlgebra, QuantEcon, StatsBase
export create_variables, linearize, num_eval, ir, simu_1st, mom,print_linearized_system  # added print_linearized_system here

mutable struct model
	f
	fy
	fx
	fyp
	fxp
	ETASHOCK
	ix
	iy
	params
	vars
end 

mutable struct model_num_eval 
	nf
	nfy 
	nfx 
	nfyp 
	nfxp 
	nETASHOCK 
	nvarshock 
	ix 
	iy 
	gx
	hx 
	exitflag 
	nx 
	ny 
	params 
	vars 
end 


"""
```
params, vars = create_variables(params_keys,vars_keys)
```
params _ keys, vars _ keys as one string

Example: params_keys = "params1, params2, params3"
"""
function create_variables(params_keys,vars_keys)
	if typeof(params_keys)==String
	    params_keys = Array{String}(strip.(split(params_keys,",")))
	end
	if typeof(vars_keys)==String
		vars_keys = Array{String}(strip.(split(vars_keys,",")))
	end 

	params = Dict{Symbol,Union{Real,Basic}}()
	vars = Dict{Symbol,Union{Real,Basic}}()
	varsp = Dict{Symbol,Union{Real,Basic}}()
    varsp_keys = vars_keys.*"p"

	@inbounds for i=1:length(params_keys)
		params[Symbol(params_keys[i])] = symbols(params_keys[i])
	end

	@inbounds for i=1:length(vars_keys)
		vars[Symbol(vars_keys[i])] = symbols(vars_keys[i])
		varsp[Symbol(varsp_keys[i])] = symbols(varsp_keys[i])
	end
	return params, vars, varsp
end

function p_it(statevar)
	return symbols.(string.(statevar).*"p")
end

"""
Computes analytical first derivatives of the function f(yp,y,xp,x) wrt x,y,xp,yp
"""
function anal_deriv(f,x,y,xp,yp; order=1)
	nx = length(x)
    ny = length(y)
    nxp = length(xp)
    nyp = length(yp)

    n = size(f,1)
    fx  = Array{Basic,2}(undef,n,nx)
	fxp = Array{Basic,2}(undef,n,nxp)
	fy  = Array{Basic,2}(undef,n,ny)
	fyp = Array{Basic,2}(undef,n,nyp)

    #Compute the first derivative of f
	@inbounds for i=1:n, j=1:nx
		fx[i,j] = diff(f[i],x[j])
		fxp[i,j] = diff(f[i],xp[j])
	end
	@inbounds for i=1:n, j=1:ny
		fy[i,j] = diff(f[i],y[j])
		fyp[i,j] = diff(f[i],yp[j])
	end
	
	return fx, fxp, fy, fyp
end

"""

"""
function print_linearized_system(f, statevar, controlvar, params, vars)
    # Get future variables
    statevarp = p_it(statevar)
    controlvarp = p_it(controlvar)
    
    # Take derivatives
    fx, fxp, fy, fyp = anal_deriv(f, statevar, controlvar, statevarp, controlvarp)
    
    println("\nLinearized system in the form: A₀yₜ + A₁yₜ₊₁ + B₀xₜ + B₁xₜ₊₁ = 0")
    println("\nWhere:")
    println("State variables (x) = ", statevar)
    println("Control variables (y) = ", controlvar)
    
    println("\nCoefficients for current states (B₀):")
    display(fx)
    
    println("\nCoefficients for future states (B₁):")
    display(fxp)
    
    println("\nCoefficients for current controls (A₀):")
    display(fy)
    
    println("\nCoefficients for future controls (A₁):")
    display(fyp)
    
    return fx, fxp, fy, fyp
end
"""

# F = linearize(f, statevar, controlvar, params, vars, shocks, STDEPS; lambdifying=true)
```
"""
function linearize(f, statevar, controlvar, params, vars, shocks, STDEPS; lambdifying=true)
	if any(f.==0.) || any(f.==0)
		throw(error("Number of equations: $(length(f[f.!=0.])) =/= Number of variables: $(length(vars))."))
	end
	if typeof(shocks)==String
		shocks = Array{String}(strip.(split(shocks,",")))
	end
	if typeof(STDEPS)==String
		STDEPS = Array{String}(strip.(split(STDEPS,",")))
	end

	statevarp = p_it(statevar)
	controlvarp = p_it(controlvar)
	cu = [statevar; controlvar];
    cup = [statevarp; controlvarp];

    #Take derivatives
    fx,fxp,fy,fyp=anal_deriv(f,statevar,controlvar,statevarp,controlvarp, order=1)

    for subcb=1:2 
        @inbounds for i=1:length(f) ,k=1:length(cup)
            f[i] = subs(f[i],(cup[k],cu[k]))
        end
        @inbounds for i=1:length(f),j=1:size(fx,2),k=1:length(cup)
            fx[i,j] = subs(fx[i,j],(cup[k],cu[k]))
            fxp[i,j] = subs(fxp[i,j],(cup[k],cu[k]))
        end
        @inbounds for i=1:length(f),j=1:size(fy,2),k=1:length(cup)
            fy[i,j] = subs(fy[i,j],(cup[k],cu[k]))
            fyp[i,j] = subs(fyp[i,j],(cup[k],cu[k]))
        end
    end
	if typeof(STDEPS)!=Basic 
		STDEPS = symbols.(STDEPS)		
	end
    nshocks = length(STDEPS);
    nstates = length(statevar);
    ETASHOCK = Array{Basic}(undef, nstates,nshocks);
    ETASHOCK[1:nstates,1:nshocks] .= 0.
    for i=1:nshocks #Make statevar a column vector!!
        shock_pos = findall(x->x==symbols(shocks[i]),statevar)[1]
        ETASHOCK[shock_pos,i] = STDEPS[i]
    end
	ix = Dict{Symbol,Int64}()
    @inbounds for i=1:length(statevar)
        ix[Symbol.(statevar[i])] = i
    end
    iy = Dict{Symbol,Int64}()
    @inbounds for i=1:length(controlvar)
        iy[Symbol.(controlvar[i])] = i
    end
	if lambdifying 
		XX = [symbols.(collect(keys(params)))...; symbols.(collect(keys(vars)))]
		f = lambdify(f,XX)
		fy = lambdify(fy,XX)
		fx = lambdify(fx,XX)
		fyp = lambdify(fyp,XX)
		fxp = lambdify(fxp,XX)
		ETASHOCK = lambdify(ETASHOCK,XX)
	end
	return model(f, fy, fx, fyp, fxp, ETASHOCK, ix, iy, params, vars)
end


"""
```
num_eval(F)
```
where F is a namedtuple that holds f,fy,fx,fyp,fxp,ETASHOCK,ix,iy,params,vars.

If invokelast error, then it means # of params and vars from steady state does not equal to # of params and vars from model file.
"""
function num_eval(F;disp=true,check=true,tol=1e-8)
	@unpack f,fy,fx,fyp,fxp,ETASHOCK,ix,iy, params,vars = F
	num_vars = [collect(values(params))...; collect(values(vars))...];
	if typeof(num_vars)==Array{Basic,1}
		throw(error("Either Parameter or Variable Value Not Assigned"));
	end
	nf = f(num_vars...)
	nfy = fy(num_vars...)
	nfx = fx(num_vars...)
	nfyp = fyp(num_vars...)
	nfxp = fxp(num_vars...)
	nETASHOCK = ETASHOCK(num_vars...)
	err = maximum(abs.(nf))<tol
	if check 
		if err ==false
			eq_ind = findall(x->x>tol,abs.(nf))
			println("Check equations ",eq_ind)
			println("Residuals ",nf[eq_ind])
			println("Check variable eq_ind for equation indices that does not satisfy the steady state conditions.")
		end
		@assert err==true "Check your steady state!"
	end 
	gx, hx, exitflag = gx_hx(nfy,nfx,nfyp,nfxp;disp=disp)
	nx = size(hx)[1]
	ny = size(gx)[1]
	nvarshock = nETASHOCK*nETASHOCK' 
	G = model_num_eval(nf, nfy, nfx, nfyp, nfxp, nETASHOCK, nvarshock, ix, iy, gx, hx, exitflag, nx, ny, params, vars)
	return G 
end

"""
```
gx,hx,exitflag = gx_hx(fy,fx,fyp,fxp;print=true)
```
Computes the matrices gx and hx that define the first-order approximation to the solution
of a dynamic stochastic general equilibrium model.

Following the notation in Schmitt-Grohe and Uribe (JEDC, 2004), the model's equilibrium conditions
take the form \n
E_t[f(yp,y,xp,x)=0. \n
The solution is of the form \n
xp = h(x,sigma) + sigma * eta * ep \n
y = g(x,sigma).

The first-order approximations to the functions g and h around the point (x,sigma)=(xbar,0), where xbar=h(xbar,0), are: \n

h(x,sigma) = xbar + hx (x-xbar)

and

g(x,sigma) = ybar + gx * (x-xbar),


where ybar=g(xbar,0).

The variable exitflag takes the values 0 (no solution), 1 (unique solution), 2 (indeterminacy), or 3 (z11 is not invertible).

Inputs: fy fyp fx fxp 

The parameter stake ensures that all eigenvalues of hx are less than stake in modulus (the default is stake=1).

Outputs: gx hx exitflag

"""
function gx_hx(fy,fx,fyp,fxp;disp=true)
    exitflag = 1

    #Creating system matrices A, B
    A = [-fxp -fyp]
    B = [fx fy]
    NK = size(fx,2)
	
    #Complex Schur Decomposition
    F = schur(complex(Matrix(A)), complex(Matrix(B)));
    s = F.S
    t = F.T
    q = F.Q
    z = F.Z

    #Pick Stable E-vals
    slt = (abs.(diag(t)).<abs.(diag(s)))
    nk = sum(slt)

    #Reorder the system with stable eigs in upper-left
    ordschur!(F,slt)

    #Split up the results appropriately
    z21 = z[nk+1:end,1:nk]
    z11 = z[1:nk,1:nk]

    s11 = s[1:nk,1:nk]
    t11 = t[1:nk,1:nk]
	if disp
		println("Number of stable eigenvalues: ", nk)
		println("Number of state variables: ", NK)
	end
    #Identify cases with no/multiple solutions
    if nk>NK
		if disp
			println("Equilibrium is locally indeterminate")
        end
		exitflag = 2
		return zeros(nk,NK),zeros(nk,NK),exitflag
    elseif nk<NK
		if disp
			println("No local equilibrium exists")
        end
		exitflag = 0
		return zeros(nk,NK),zeros(nk,NK),exitflag
    end

    if rank(z11)<nk
		if disp
			println("Rank of z11: $(rank(z11)).") 
			println("Number of stable eigenvalues: ", nk) 
			println("Invertibility condition violated")
        end
		exitflag = 3
		return zeros(2),zeros(2),exitflag
    end

    #Compute solution
    z11i = z11\I
    gx = real(z21*z11i)
    hx = real(z11*(s11\t11)*z11i)

    return gx, hx, exitflag
end


"""
Get the unconditional variance-covariance matrix of x and y using doubling algorithm

Input: Gx(obs), Hx(state), nvarshock,

output: sigyJ, sigxJ

optional: J = integer, J=0 is the default (asymptotic moments)

method = 1, uses kronecker algorithm to calculate moments.
0, which uses doubling, is default
"""
function mom(gx, hx, varshock; J=0,method=0)
	sigx = zeros(size(hx));
    if method == 0
		hx_old = hx
		sig_old=varshock
		sigx_old = Matrix{Float64}(I,size(hx))
		diferenz=0.1
		sigx = Array{Float64}(undef,size(sig_old))
		while diferenz>1e-25
			sigx = hx_old*sigx_old*hx_old'+sig_old
			diferenz = findmax(abs.(sigx.-sigx_old))[1]; #calc_dc(sigx,sigx_old)
			sig_old = hx_old*sig_old*hx_old'+sig_old
			hx_old=hx_old*hx_old
			sigx_old = sigx[:,:]
		end
	else
		sigx = zeros(size(hx));
		F = kron(hx,hx);
		sigx[:] = (I-F)\varshock[:];
	end

    sigxJ=hx^(-minimum((0,J)))*sigx*(hx')^(maximum((0,J)))

    sigyJ=real(gx*sigxJ*gx')
    sigxJ=real(sigxJ)

    return sigyJ, sigxJ
end

"""
```
IR, IRy, IRx  = ir(gx, hx, x0, T)
```

example.

x0 = zeros(length(ns))

x0[end] = 1.

IR,IRy,IRx = ir(gx,hx,x0,T)

Computes T-period impulse responses (3/15/2019)

Inputs: gx, hx, x0 (nx vector), T

Outputs: IR, IRy, IRx

"""
function ir(gx, hx, x0, T)
    x0 = x0[:]
    pd = length(x0)
    MX = [gx;  Matrix{Float64}(I, pd, pd)]
    IR = Array{Float64}(undef,T,size(MX)[1])
    x = x0
    for t=1:T
        IR[t,:] = (MX*x)'
        x = hx * x
    end

    IRx = IR[:,end-pd+1:end]
    IRy = IR[:,1:end-pd]
    return IR, IRy, IRx
end


"""
```
X, Y = simu_1st(gx,hx,eta,T)

X, Y = simu_1st(gx,hx,eta,T,e)
```
Simulates time series from the model \n
x_t+1 = hx x_t + eta e_t+1 for t=1,...,T-1 \n
Y_t = gx * x_t for t=1,...,T  \n

Inputs are gx, hx, netashock,T, x0, e \n
hx is nx by nx \n
gx is ny by nx \n
eta is nx by ne \n (If shocks are correlated, eta = correl*sigma)
e is T by ne random shock \n
T is the length of the simulation \n
x0 is the initial condition for X \n

Outputs are X (T by nx) and Y (T by ny)
"""
function simu_1st(gx,hx,eta,T; set_seed=true)
	if set_seed 
		Random.seed!(1)
	end
	nx = size(hx,1)
	ny = size(gx,1)
	ne = size(eta,2)

	#Initialize X and Y
	x = zeros(T,nx)
	y = zeros(T,ny)
	x0 = zeros(nx)
	e = randn(T,ne);

	x[1,1:nx] .= x0
	y[1,1:ny] .= gx*x[1,:]

	for t=2:T
		x[t,:] .= vec(hx*x[t-1,:] .+ eta*e[t,:]);
		y[t,1:ny] .= gx*x[t,:];
	end
    return x, y
end


end 
