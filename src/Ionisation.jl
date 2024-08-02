module Ionisation
import SpecialFunctions: gamma, dawson, ellipk, ellipe
import GSL: hypergeom
import HDF5
import FileWatching.Pidfile: mkpidlock
import Logging: @info, @warn
import Luna.PhysData: c, ħ, μ_0, electron, m_e, au_energy, au_time, au_Efield, wlfreq
import Luna.PhysData: ionisation_potential, quantum_numbers
import Luna: Maths, Utils
import Printf: @sprintf

"""
    ionrate_fun!_ADK(ionpot::Float64, threshold=true)
    ionrate_fun!_ADK(material::Symbol)

Return a closure `ionrate!(out, E)` which calculates the ADK ionisation rate for the electric
field `E` and places the result in `out`. If `threshold` is true, use [`ADK_threshold`](@ref)
to avoid calculation below floating-point precision. If `cycle_average` is `true`, calculate
the cycle-averaged ADK ionisation rate instead.
"""
function ionrate_fun!_ADK(ionpot::Float64, threshold=true; cycle_average=false)
    nstar = sqrt(0.5/(ionpot/au_energy))
    cn_sq = 2^(2*nstar)/(nstar*gamma(nstar+1)*gamma(nstar))
    ω_p = ionpot/ħ
    ω_t_prefac = electron/sqrt(2*m_e*ionpot)

    if threshold
        thr = ADK_threshold(ionpot)
    else
        thr = 0
    end

    # Zenghu Chang: Fundamentals of Attosecond Optics (2011) p. 184
    # Section 4.2.3.1 Cycle-Averaged Rate
    # ̄w_ADK(Fₐ) = √(3/π) √(Fₐ/F₀) w_ADK(Fₐ) where Fₐ is the field amplitude 
    Ip_au = ionpot / au_energy
    F0_au = (2Ip_au)^(3/2)
    F0 = F0_au*au_Efield
    avfac = sqrt.(3/(π*F0))


    ionrate! = let nstar=nstar, cn_sq=cn_sq, ω_p=ω_p, ω_t_prefac=ω_t_prefac, thr=thr
        function ir(E)
            if abs(E) >= thr
                r = (ω_p*cn_sq*
                    (4*ω_p/(ω_t_prefac*abs(E)))^(2*nstar-1)
                    *exp(-4/3*ω_p/(ω_t_prefac*abs(E))))
                if cycle_average
                    r *= avfac*sqrt(abs(E))
                end
                return r
            else
                return zero(E)
            end
        end
        function ionrate!(out, E)
            out .= ir.(E)
        end
    end

    return ionrate!  
end

function ionrate_fun!_ADK(material::Symbol; kwargs...)
    return ionrate_fun!_ADK(ionisation_potential(material); kwargs...)
end

function ionrate_ADK(IP_or_material, E; kwargs...)
    out = zero(E)
    ionrate_fun!_ADK(IP_or_material; kwargs...)(out, E)
    return out
end

function ionrate_ADK(IP_or_material, E::Number; kwargs...)
    out = [zero(E)]
    ionrate_fun!_ADK(IP_or_material; kwargs...)(out, [E])
    return out[1]
end

"""
    ADK_threshold(ionpot)

Determine the lowest electric field strength at which the ADK ionisation rate for the
ionisation potential `ionpot` is non-zero to within 64-bit floating-point precision.
"""
function ADK_threshold(ionpot)
    out = [0.0]
    ADKfun = ionrate_fun!_ADK(ionpot, false)
    E = 1e3
    while out[1] == 0
        E *= 1.01
        ADKfun(out, [E])
    end
    return E
end

"""
    ionrate_fun!_PPTaccel(material::Symbol, λ0; kwargs...)
    ionrate_fun!_PPTaccel(ionpot::Float64, λ0, Z, l; kwargs...)

Create an accelerated (interpolated) PPT ionisation rate function.
"""
function ionrate_fun!_PPTaccel(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    ionrate_fun!_PPTaccel(ip, λ0, Z, l; kwargs...)
end

function ionrate_fun!_PPTaccel(ionpot::Float64, λ0, Z, l; kwargs...)
    E, rate = makePPTcache(ionpot, λ0, Z, l; kwargs...)
    return makeAccel(E, rate)
end

"""
    ionrate_fun!_PPTcached(material::Symbol, λ0; kwargs...)
    ionrate_fun!_PPTcached(ionpot::Float64, λ0, Z, l; kwargs...)

Create a cached (saved) interpolated PPT ionisation rate function. If a saved lookup table
exists, load this rather than recalculate.

# Keyword arguments
- `N::Int`: Number of samples with which to create the `CSpline` interpolant.
- `Emax::Number`: Maximum field strength to include in the interpolant.
- `cachedir::String`: Path to the directory where the cache should be stored and loaded from.
    Defaults to \$HOME/.luna/pptcache

Other keyword arguments are passed on to [`ionrate_fun_PPT`](@ref)
"""
function ionrate_fun!_PPTcached(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    ionrate_fun!_PPTcached(ip, λ0, Z, l; kwargs...)
end

function ionrate_fun!_PPTcached(ionpot::Float64, λ0, Z, l;
                                sum_tol=1e-4, cycle_average=false, N=2^16, Emax=nothing,
                                cachedir=joinpath(Utils.cachedir(), "pptcache"),
                                stale_age=60*10)
    h = hash((ionpot, λ0, Z, l, sum_tol, cycle_average, N, Emax))
    fname = string(h, base=16)*".h5"
    fpath = joinpath(cachedir, fname)
    lockpath = joinpath(cachedir, "pptlock")
    isdir(cachedir) || mkpath(cachedir)
    if isfile(fpath)
        @info @sprintf("Found cached PPT rate for %.2f eV, %.1f nm", ionpot/electron, 1e9λ0)
        rate = mkpidlock(lockpath; stale_age) do
            loadPPTaccel(fpath)
        end
        return rate
    else
        E, rate = makePPTcache(ionpot::Float64, λ0, Z, l;
                               sum_tol=sum_tol, cycle_average, N=N, Emax=Emax)
        mkpidlock(lockpath; stale_age) do
            if ~isfile(fpath) # makePPTcache takes a while - has another process saved first?
                @info @sprintf(
                    "Saving PPT rate for %.2f eV, %.1f nm in %s",
                    ionpot/electron, 1e9λ0, cachedir
                )
                HDF5.h5open(fpath, "cw") do file
                    file["E"] = E
                    file["rate"] = rate
                end
            end
        end
        return makeAccel(E, rate)
    end
end

function loadPPTaccel(fpath)
    isfile(fpath) || error("PPT cache file $fpath not found!")
    E, rate = HDF5.h5open(fpath, "r") do file
        (read(file["E"]), read(file["rate"]))
    end
    makeAccel(E, rate)
end

function makePPTcache(ionpot::Float64, λ0, Z, l;
                      sum_tol=1e-4, cycle_average=false, N=2^16, Emax=nothing)
    Emax = isnothing(Emax) ? 2*barrier_suppression(ionpot, Z) : Emax

    # ω0 = 2π*c/λ0
    # Emin = ω0*sqrt(2m_e*ionpot)/electron/0.5 # Keldysh parameter of 0.5
    Emin = Emax/5000

    E = collect(range(Emin, stop=Emax, length=N));
    @info @sprintf("Pre-calculating PPT rate rate for %.2f eV, %.1f nm...", ionpot/electron, 1e9λ0)
    rate = ionrate_PPT(ionpot, λ0, Z, l, E; sum_tol=sum_tol, cycle_average);
    @info "...PPT pre-calculation done"
    return E, rate
end

"""
    barrier_suppression(ionpot, Z)

Calculate the barrier-suppresion **field strength** for the ionisation potential `ionpot`
and charge state `Z`.
"""
function barrier_suppression(ionpot, Z)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2*Ip_au)
    Z^3/(16*ns^4) * au_Efield
end

"""
    keldysh(material, λ, E)

Calculate the Keldysh parameter for the given `material` at wavelength `λ` and electric field
strength `E`.
"""
function keldysh(material, λ, E)
    Ip_au = ionisation_potential(material)/au_energy
    E_au = E/au_Efield
    ω0_au = wlfreq(λ)*au_time
    ω0_au*sqrt(2Ip_au)/E_au
end

"""
    ionfrac(rate, E, δt)

Given an ionisation rate function `rate` and an electric field array `E` sampled with time
spacing `δt`, calculate the ionisation fraction as a function of time on the same time axis.

The function `rate` should have the signature `rate!(out, E)` and place its results into
`out`, like the functions returned by e.g. `ionrate_fun!_ADK` or `ionrate_fun!_PPTcached`.
"""
function ionfrac(rate, E, δt)
    frac = similar(E)
    ionfrac!(frac, rate, E, δt)
end

function ionfrac!(frac, rate, E, δt)
    rate(frac, E)
    Maths.cumtrapz!(frac, δt)
    @. frac = 1 - exp(-frac)
end

function makeAccel(E, rate)
    # Interpolating the log and re-exponentiating makes the spline more accurate
    cspl = Maths.CSpline(E, log.(rate); bounds_error=true)
    Emin = minimum(E)
    Emax = maximum(E)
    function ionrate!(out, E)
        for ii in eachindex(out)
            aE = abs(E[ii])
            if aE < Emin
                out[ii] = 0.0
            elseif aE > Emax
                error(
                    "Field strength $aE V/m exceeds maximum for ionisation rate ($Emax V/m)."
                    )
            else
                out[ii] = exp(cspl(aE))
            end
        end
    end
end

function ionrate_fun!_PPT(args...)
    ir = ionrate_fun_PPT(args...)
    function ionrate!(out, E)
        out .= ir.(E)
    end
    return ionrate!
end

"""
    ionrate_fun_PPT(ionpot::Float64, λ0, Z, l; sum_tol=1e-4, cycle_average=false)

Create closure to calculate PPT ionisation rate.

# Keyword arguments
- `sum_tol::Number`: Relative tolerance used to truncate the infinite sum.
- `cycle_average::Bool`: If `false` (default), calculate the cycle-averaged rate

# References
[1] Ilkov, F. A., Decker, J. E. & Chin, S. L.
Ionization of atoms in the tunnelling regime with experimental evidence
using Hg atoms. Journal of Physics B: Atomic, Molecular and Optical
Physics 25, 4005–4020 (1992)

[2] 1.Bergé, L., Skupin, S., Nuter, R., Kasparian, J. & Wolf, J.-P.
Ultrashort filaments of light in weakly ionized, optically transparent
media. Rep. Prog. Phys. 70, 1633–1713 (2007)
(Appendix A)
"""
function ionrate_fun_PPT(ionpot::Float64, λ0, Z, l; sum_tol=1e-4, cycle_average=false)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2Ip_au)
    ls = ns-1
    Cnl2 = 2^(2ns)/(ns*gamma(ns + ls + 1)*gamma(ns - ls))

    ω0 = 2π*c/λ0
    ω0_au = au_time*ω0
    E0_au = (2*Ip_au)^(3/2)

    ionrate = let ω0_au=ω0_au, Cnl2=Cnl2, ns=ns, sum_tol=sum_tol
        function ionrate(E)
            E_au = abs(E)/au_Efield
            γ = ω0_au*sqrt(2Ip_au)/E_au
            γ2 = γ*γ
            β = 2γ/sqrt(1 + γ2)
            α = 2*(asinh(γ) - γ/sqrt(1+γ2))
            Up_au = E_au^2/(4*ω0_au^2)
            Uit_au = Ip_au + Up_au
            v = Uit_au/ω0_au
            ret = 0
            divider = 0
            for m = -l:l
                divider += 1
                mabs = abs(m)
                flm = ((2l + 1)*factorial(l + mabs)
                    / (2^mabs*factorial(mabs)*factorial(l - mabs)))
                # Following 5 lines are [1] eq. 8 and lead to identical results:
                # G = 3/(2γ)*((1 + 1/(2γ2))*asinh(γ) - sqrt(1 + γ2)/(2γ))
                # Am = 4/(sqrt(3π)*factorial(mabs))*γ2/(1 + γ2)
                # lret = sqrt(3/(2π))*Cnl2*flm*Ip_au
                # lret *= (2*E0_au/(E_au*sqrt(1 + γ2))) ^ (2ns - mabs - 3/2)
                # lret *= Am*exp(-2*E0_au*G/(3E_au))
                # [2] eq. (A14) 
                lret = 4sqrt(2)/π*Cnl2
                lret *= (2*E0_au/(E_au*sqrt(1 + γ2))) ^ (2ns - mabs - 3/2)
                lret *= flm/factorial(mabs)
                lret *= exp(-2v*(asinh(γ) - γ*sqrt(1+γ2)/(1+2γ2)))
                lret *= Ip_au * γ2/(1+γ2)
                # Remove cycle average factor, see eq. (2) of [1]
                if !cycle_average
                    lret *= sqrt(π*E0_au/(3E_au))
                end
                k = ceil(v)
                n0 = ceil(v)
                sumfunc = let k=k, β=β, m=m
                    function sumfunc(x, n)
                        diff = n-v
                        return x + exp(-α*diff)*φ(m, sqrt(β*diff))
                    end
                end
                # s, success, steps = Maths.aitken_accelerate(
                #     sumfunc, 0, n0=n0, rtol=sum_tol, maxiter=Inf)
                s, success, steps = Maths.converge_series(
                    sumfunc, 0, n0=n0, rtol=sum_tol, maxiter=Inf)
                lret *= s
                ret += lret
            end
            return ret/(au_time*divider)
        end
    end
    return ionrate
end

"""
    φ(m, x)

Calculate the φ function for the PPT ionisation rate.

Note that w_m(x) in [1] and φ_m(x) in [2] look slightly different but
are in fact identical.
"""
function φ(m, x)
    mabs = abs(m)
    return (exp(-x^2)
            * sqrt(π)
            * x^(mabs+1)
            * gamma(mabs+1)
            * hypergeom(1/2, 3/2 + mabs, x^2)
            / (2*gamma(3/2 + mabs)))
end

function ionrate_fun_PPT(material::Symbol, λ0; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_fun_PPT(ip, λ0, Z, l; kwargs...)
end

function ionrate_PPT(ionpot, λ0, Z, l, E; kwargs...)
    return ionrate_fun_PPT(ionpot, λ0, Z, l; kwargs...).(E)
end

function ionrate_PPT(material::Symbol, λ0, E; kwargs...)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_PPT(ip, λ0, Z, l, E; kwargs...)
end






#--------------------------------------------------------------------------------------~
"""
    ionrate_fun!_Keldyshaccel(material::Symbol, λ0; kwargs...)
    ionrate_fun!_Keldyshaccel(ionpot::Float64, λ0, Z, l; kwargs...)

Create an accelerated (interpolated) Keldysh ionisation rate function.
"""
function ionrate_fun!_keldysaccel(material::Symbol, λ0; kwargs...)
    ip = ionisation_potential(material)
    ionrate_fun!_Keldyshaccel(ip, λ0; kwargs...)
end

function ionrate_fun!_Keldyshaccel(ionpot::Float64, λ0; kwargs...)
    E, rate = makeKeldyshcache(ionpot, λ0; kwargs...)
    return makeAccel(E, rate)
end

"""
    ionrate_fun!_Keldyshcached(material::Symbol, λ0; kwargs...)
    ionrate_fun!_Keldyshcached(ionpot::Float64, λ0, Z, l; kwargs...)

Create a cached (saved) interpolated Keldysh ionisation rate function. If a saved lookup table
exists, load this rather than recalculate.

# Keyword arguments
- `N::Int`: Number of samples with which to create the `CSpline` interpolant.
- `Emax::Number`: Maximum field strength to include in the interpolant.
- `cachedir::String`: Path to the directory where the cache should be stored and loaded from.
    Defaults to \$HOME/.luna/keldyshcache

Other keyword arguments are passed on to [`ionrate_fun_Keldysh`](@ref)
"""
function ionrate_fun!_Keldyshcached(material::Symbol, λ0; kwargs...)
    ip = ionisation_potential(material)
    ionrate_fun!_Keldyshcached(ip, λ0; kwargs...)
end

function ionrate_fun!_Keldyshcached(ionpot::Float64, λ0;
                                 rtol = 1e-6, maxiter = 10000, N=2^16, Emax=nothing,
                                cachedir=joinpath(Utils.cachedir(), "keldyshcache"),
                                stale_age=60*10)
    h = hash((ionpot, λ0, rtol, maxiter, N, Emax))
    fname = string(h, base=16)*".h5"
    fpath = joinpath(cachedir, fname)
    print(fpath, "\n")
    lockpath = joinpath(cachedir, "keldyshlock")
    isdir(cachedir) || mkpath(cachedir)
    if isfile(fpath)
        @info @sprintf("Found cached Keldysh rate for %.2f eV, %.1f nm", ionpot/electron, 1e9λ0)
        rate = mkpidlock(lockpath; stale_age) do
            loadKeldyshaccel(fpath)
        end
        return rate
    else
        E, rate = makeKeldyshcache(ionpot::Float64, λ0;
                                rtol = rtol, maxiter = maxiter, N=N, Emax=Emax)
        mkpidlock(lockpath; stale_age) do
            if ~isfile(fpath) # makeKeldyshcache takes a while - has another process saved first?
                @info @sprintf(
                    "Saving Keldysh rate for %.2f eV, %.1f nm in %s",
                    ionpot/electron, 1e9λ0, cachedir
                )
                HDF5.h5open(fpath, "cw") do file
                    file["E"] = E
                    file["rate"] = rate
                end
            end
        end
        return makeAccel(E, rate)
    end
end

function loadKeldyshaccel(fpath)
    isfile(fpath) || error("Keldysh cache file $fpath not found!")
    E, rate = HDF5.h5open(fpath, "r") do file
        (read(file["E"]), read(file["rate"]))
    end
    makeAccel(E, rate)
end

function makeKeldyshcache(ionpot::Float64, λ0;
                        rtol = 1e-6, maxiter = 10000, N=2^16, Emax=nothing)
    Imax=ionpot/λ0^2 #Intensity in W/cm^2 where the energy matches the ionisation potential and area is the mininum of λ^2
    @info @sprintf("Imax=%.2e", Imax)
    # Emax = isnothing(Emax) ? 1000*sqrt(μ_0*c*Imax) : Emax
    Emax = isnothing(Emax) ? 1e10 : Emax
    @info @sprintf("Emax=%.2e", Emax)
    # ω0 = 2π*c/λ0
    # Emin = ω0*sqrt(2m_e*ionpot)/electron/0.5 # Keldysh parameter of 0.5
    Emin = Emax/5000

    E = collect(range(Emin, stop=Emax, length=N));
    @info @sprintf("Pre-calculating Keldysh rate rate for %.2f eV, %.1f nm...", ionpot/electron, 1e9λ0)
    rate = ionrate_Keldysh(ionpot, λ0, E; rtol = rtol, maxiter = maxiter);
    @info "...Keldysh pre-calculation done"
    return E, rate
end


"""
    ionrate_fun_keldysh(ionpot::Float64, λ0; Nsum=1e3)

Create closure to calculate Keldysh ionisation rate.
# Keyword arguments
- `rtol::Number`: Relative tolerance to stop the calculation of the sumation
- `maxiter::Number`: Maximum number of iteration to calculate the sumation
the variable Q(γ,x) [1]

# References
[1] COUAIRON, A., & MYSYROWICZ, A.  
Femtosecond filamentation in transparent media.
 Physics Reports, 441(2–4), 47–189. (2007).
 page 85

[2] Majus, D., Jukna, V., Tamošauskas, G., Valiulis, G., & Dubietis, A. 
Three-dimensional mapping of multiple filament arrays. 
Physical Review A, 81(4), 043811. (2010). 

[3] Keldysh, L. v. (2023). Ionization in the field of a strong 
electromagnetic wave. In Selected Papers of Leonid V Keldysh 
(Vol. 20, Issue 5, pp. 56–63).
"""


function ionrate_fun!_Keldysh(ionpot::Float64, λ0; rtol = 1e-6, maxiter = 10000)
    #=After testing the units of te expression seem to be SI and not atomic. 
    Atomic gives extremelly small rates. Check if it is beacuse of a different factor=#
    Ip_au = ionpot# / au_energy #bangap
    ω0 = 2π*c/λ0
    ω0_au = ω0#*au_time #central frequency
    m_au=0.635*m_e #usual reduced electron-hole mass is 0.635*m_e [2]  page 3. Which in atomic units is just 0.635


    ionrate! = let ω0_au=ω0_au, m_au=m_au, Ip_au=Ip_au, rtol = rtol, maxiter = maxiter
        function ir(E) 
            
            E_au = abs(E)#/au_Efield
            if E_au==0
                return 0
            end
            if isnan(E_au)
                ArgumentError("The electric field is NaN")
            end

            # print(E_au)

            # Eq. 91 [1]
            γ = ω0_au/electron/E_au*sqrt(m_au*Ip_au)
            # γ = ω0_au/E_au*sqrt(m_au*Ip_au) #in thse units the electron=1
            # @info @sprintf("E_au=%.3e, ω0_au=%.3e, m_au=%.3e, Ip_au=%.3e, γ=%.3e",E_au, ω0_au, m_au, Ip_au, γ)

            # Eq. 93 [1]
            Γ=γ^2/(1+γ^2)
            Ξ=1/(1+γ^2)
            
            #=
            Complete elliptical integrals
            The square factor in the following is due to the definition of the elliptical intergrals of SpecialFunctions.jl.
            https://specialfunctions.juliamath.org/latest/functions_list/#SpecialFunctions.ellipk-Tuple{Real}
            They use the m factor while the original equations by Keldysh [3] use the k^2 definition mentioned in the SpecialFunctions.jl.
            An easy way to test it is using the Legendre identity mentioned in [3] as a un-numbered expression between Eq. 39 and 40.
            x=0.1 #xϵ[-Inf,1]
            sx=sqrt(1-x^2)
            K(x)*E(sx)+K(sx)*E(x)-K(x)*K(sx)-pi/2 #Pure from Julia. Nonzero.
            K(x^2)*E(sx^2)+K(sx^2)*E(x^2)-K(x^2)*K(sx^2)-pi/2 #Using the k^2 as Keldysh. Zero proving the identity.
            =#
            KΓ=ellipk(Γ^2)
            if isinf(KΓ)
                #=
                When KΓ→∞, α→∞ meaning that Eq 94 is reduced to its 1st order term (or goes to 0 depending on how you want to handel n=0).
                Regardless the ionisation rate in Eq. 92 goes to 0.
                Physically this is the case of low electric fields that do not trigger any ionisation as KΓ→∞ ⇄ E→0.
                =#
                return 0.0                
            end
            KΞ=ellipk(Ξ^2)
            EΓ=ellipe(Γ^2)
            EΞ=ellipe(Ξ^2)

            # @info @sprintf("Γ=%.2e, Ξ=%.2e, KΓ=%.2e, KΞ=%.2e, EΓ=%.2e, EΞ=%.2e",Γ,Ξ,KΓ,KΞ,EΓ,EΞ)

            # Eq. 95 [1]
            α=π*(KΓ-EΓ)/EΞ
            β=π^2/(4*KΞ*EΞ)
            
            # Eq. 96 [1]
            x=2/π*Ip_au/ħ/ω0_au*EΞ/sqrt(Γ)
            # x=2/π*Ip_au/ω0_au*EΞ/sqrt(Γ) #In these units ħ=1
            ν=floor(x+1, digits=0)-x #Check if it makes sense with these units
            
            # Eq. 94 [1]
            f(n)=exp(-n*α)*dawson(sqrt(β*(n+2.0*ν)))
            result=Maths.converge_sum(f, n0 = 0, rtol = 1e-6, maxiter = 10000)
            if result[2]== false 
                @warn "Failed to converge sum during calculation of term Q with  rtol = $rtol, maxiter = $maxiter." maxlog=1
            end
            Q=sqrt(π/2/KΞ)*result[1]

            #=
            ALTERNATIVE THE HANDELING for K(Γ) MENTIONED ABOVE.
            USED TO TEST THE VALIDITY AND IT IS THE SAME

            A simplistic way  to deal with infinity in the K(Γ).
            From the equations 95 in [1] when K(Γ)->∞ → α->∞.
            This means than in Eq 94 the only contribution to Q is n=0.
            Physically it makes sense since K(Γ)->∞ when Γ->1 ↔ γ->∞ ↔ E->0. 
            =#
            # if ~isinf(KΓ)
            #     f(n)=exp(-n*α)*dawson(sqrt(β*(n+2.0*ν)))
            #     result=Maths.converge_sum(f, n0 = 0, rtol = 1e-6, maxiter = 10000)
            #     if result[2]== false 
            #         @warn "Failed to converge sum during calculation of term Q with  rtol = $rtol, maxiter = $maxiter."
            #     end
            #     Q=sqrt(π/2/KΞ)*result[1]
            #     # @info @sprintf("ΔQ=%.3e",1-sqrt(π/2/KΞ)*dawson(sqrt(β*(2.0*ν)))/Q)
            # else
            #     Q=sqrt(π/2/KΞ)*dawson(sqrt(β*(2.0*ν)))
            # end


            # @info @sprintf("α=%.2e, β=%.2e, x=%.2e, ν=%.2e, Q=%.2e", α, β, x, ν, Q)

            # Eq. 92 [1]
            ret=2*ω0_au/9/π*(ω0_au*m_au/ħ/sqrt(Γ))^1.5*Q*exp(-α*floor(x+1, digits=0))
            # ret=2*ω0_au/9/π*(ω0_au*m_au/sqrt(Γ))^1.5*Q*exp(-α*floor(x+1, digits=0)) #In these units ħ=1
            #same as ν

            
            return ret#*au_time #Reconvert to SI
        end
        function ionrate!(out, E)
            # @info out
            # @info E
            # @info ir.(E)
            out .= ir.(E)
        end
    end
    return ionrate!
end


function ionrate_fun!_Keldysh(material::Symbol, λ0; kwargs...)
    return ionrate_fun!_Keldysh(ionisation_potential(material), λ0; kwargs...)
end

function ionrate_Keldysh(IP_or_material, λ0, E; kwargs...)
    out = zero(E)
    ionrate_fun!_Keldysh(IP_or_material, λ0; kwargs...)(out, E)
    return out
end

function ionrate_Keldysh(IP_or_material, λ0, E::Number; kwargs...)
    out = [zero(E)]
    ionrate_fun!_Keldysh(IP_or_material, λ0; kwargs...)(out, [E])
    return out[1]
end



end