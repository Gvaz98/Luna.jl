module Luna
import FFTW
import Hankel
import Logging
import LinearAlgebra: mul!, ldiv!
import Random: MersenneTwister

"""
    HDF5LOCK

Lock on the HDF5 library for multi-threaded execution.
"""
const HDF5LOCK = ReentrantLock()

"""
    @hlock

Wait for HDF5LOCK, execute the expression, and release H5DFLOCK.

!!! warning
    For thread safety, any call to functions from HDF5.jl needs to be preceeded by @hlock.
"""
macro hlock(expr)
    quote
        try
            lock(HDF5LOCK)
            $(esc(expr))
        finally
            unlock(HDF5LOCK)
        end
    end
end

"""
    Luna.settings

Dictionary of global settings for `Luna`.
"""
settings = Dict{String, Any}("fftw_flag" => FFTW.PATIENT)

"""
    set_fftw_mode(mode)

Set FFTW planning mode for all FFTW transform planning in `Luna`.

Possible values for `mode` are `:estimate`, `:measure`, `:patient`, and `:exhaustive`.
The initial value upon loading `Luna` is `:patient`

# Examples
```jldoctest
julia> Luna.set_fftw_mode(:patient)
0x00000020
```
"""
function set_fftw_mode(mode)
    s = uppercase(string(mode))
    flag = getfield(FFTW, Symbol(s))
    settings["fftw_flag"] = flag
end

include("Utils.jl")
include("Scans.jl")
include("Output.jl")
include("Maths.jl")
include("PhysData.jl")
include("Grid.jl")
include("RK45.jl")
include("Modes.jl")
include("Capillary.jl")
include("RectModes.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("NonlinearRHS.jl")
include("LinearOps.jl")
include("Stats.jl")
include("Polarisation.jl")
include("Tools.jl")
include("Plotting.jl")
include("Raman.jl")
include("Field.jl")

function setup(grid::Grid.RealGrid, densityfun, normfun, responses, inputs, aeff)
    Utils.loadFFTwisdom()
    xo = Array{Float64}(undef, length(grid.to))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, normfun, aeff)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    Eω = make_init(grid, inputs, Field.energy_modal(), FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, densityfun, normfun, responses, inputs, aeff)
    Utils.loadFFTwisdom()
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, normfun, aeff)
    Eω = make_init(grid, inputs, Field.energy_modal(), FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

# for multimode setup, inputs is a tuple of ((mode_index, inputs), (mode_index, inputs), ..)
function setup(grid::Grid.RealGrid, densityfun, normfun, responses, inputs,
               modes, components; full=false)
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], Field.energy_modal(), FTt)
    end
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    xo = Array{Float64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

# for multimode setup, inputs is a tuple of ((mode_index, inputs), (mode_index, inputs), ..)
function setup(grid::Grid.EnvGrid, densityfun, normfun, responses, inputs,
               modes, components; full=false)
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    for i in 1:length(inputs)
        Eω[:,inputs[i][1]] .= make_init(grid, inputs[i][2], Field.energy_modal(), FTt)
    end
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, normfun,
                                 rtol=1e-3, atol=0.0, mfcn=300, full=full)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eω, transform, FT
end

function setup(grid::Grid.RealGrid, q::Hankel.QDHT,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xt = zeros(Float64, length(grid.t), length(q.r))
    FT = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    energy_t = Field.energy_radial(grid, q)[1]
    for input! in inputs
        input!(Eω, grid, energy_t, FT)
    end
    Eωk = q * Eω
    xo = Array{Float64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, q::Hankel.QDHT,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    xt = zeros(ComplexF64, length(grid.t), length(q.r))
    FT = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    energy_t = Field.energy_radial(grid, q)[1]
    for input! in inputs
        input!(Eω, grid, energy_t, FT)
    end
    Eωk = q * Eω
    xo = Array{ComplexF64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.RealGrid, xygrid::Grid.FreeGrid,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y          
    xr = Array{Float64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    energy_t = Field.energy_free(grid, xygrid)[1]
    for input! in inputs
        input!(Eωk, grid, energy_t, FT)
    end
    xo = Array{Float64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_rfft(xo, (1, 2, 3), flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransFree(grid, FTo, length(y), length(x),
                                       responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
               densityfun, normfun, responses, inputs)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y          
    xr = Array{ComplexF64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    energy_t = Field.energy_free(grid, xygrid)[1]
    for input! in inputs
        input!(Eωk, grid, energy_t, FT)
    end
    xo = Array{ComplexF64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_fft(xo, (1, 2, 3), flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransFree(grid, FTo, length(y), length(x),
                                       responses, densityfun, normfun)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Eωk, transform, FT
end

function make_init(grid, inputs, energy_t, FT)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for input! in inputs
        input!(out, grid, energy_t, FT)
    end
    return out
end

function run(Eω, grid,
             linop, transform, FT, output;
             min_dz=0, max_dz=Inf, init_dz=1e-4,
             rtol=1e-6, atol=1e-10, safety=0.9, norm=RK45.weaknorm,
             status_period=1)


    Et = FT \ Eω

    z = 0.0

    window! = let window=grid.ωwin, twindow=grid.twin, FT=FT, Et=Et
        function window!(Eω)
            Eω .*= window
            ldiv!(Et, FT, Eω)
            Et .*= twindow
            mul!(Eω, FT, Et)
        end
    end

    function stepfun(Eω, z, dz, interpolant)
        window!(Eω)
        output(Eω, z, dz, interpolant)
    end

    output(Grid.to_dict(grid), group="grid")

    RK45.solve_precon(
        transform, linop, Eω, z, init_dz, grid.zmax, stepfun=stepfun,
        max_dt=max_dz, min_dt=min_dz,
        rtol=rtol, atol=atol, safety=safety, norm=norm,
        status_period=status_period)
end

end # module
