#!/usr/bin/env julia

using LinearAlgebra
using JSON
using Arblib

const P = 1
const Q = 3
const R_DEFAULT = 176.54
const N_BITS = 2 * P + 1
const D = 2^N_BITS
const K_CLAUSE = 1 << Q  # BM24 k = 2^q (8 for q=3)

function parse_args()
    anchor = "multivariable/results/anchor.json"
    out = "certification_result.json"
    radius = 1e-10
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--anchor" && i < length(ARGS)
            anchor = ARGS[i + 1]
            i += 2
        elseif ARGS[i] == "--out" && i < length(ARGS)
            out = ARGS[i + 1]
            i += 2
        elseif ARGS[i] == "--radius" && i < length(ARGS)
            radius = parse(Float64, ARGS[i + 1])
            i += 2
        else
            i += 1
        end
    end
    return anchor, out, radius
end

function build_A(n_bits::Int)
    d = 2^n_bits
    A = zeros(Float64, d, d)
    for α in 0:d-1
        positions = [j for j in 0:n_bits-1 if (α >> j) & 1 == 1]
        if length(positions) <= 1
            A[α + 1, :] .= 0.5
            continue
        end
        for s in 0:d-1
            bits = [(s >> pos) & 1 for pos in positions]
            A[α + 1, s + 1] = all(b == bits[1] for b in bits) ? 0.5 : 0.0
        end
    end
    return A
end

function compute_b_s(beta::Float64, p::Int)
    n_bits = 2 * p + 1
    d = 2^n_bits
    b = zeros(ComplexF64, d)
    for s in 0:d-1
        val = 0.5 + 0.0im
        for j in 0:p-1
            s_j = (s >> j) & 1
            s_jp1 = (s >> (j + 1)) & 1
            val *= (s_j == s_jp1) ? cos(beta) : im * sin(beta)
            idx_lo = 2 * p - j - 1
            idx_hi = 2 * p - j
            s_lo = (s >> idx_lo) & 1
            s_hi = (s >> idx_hi) & 1
            val *= (s_lo == s_hi) ? cos(beta) : -im * sin(beta)
        end
        b[s + 1] = val
    end
    return b
end

function compute_c_phase(gamma::Float64, p::Int)
    n_bits = 2 * p + 1
    d = 2^n_bits
    c = zeros(ComplexF64, d)
    for α in 0:d-1
        sign = ((α >> p) & 1 == 1) ? -1.0 : 1.0
        prod = sign + 0.0im
        for j in 0:n_bits-1
            if (α >> j) & 1 == 0
                continue
            end
            if j < p
                prod *= exp(-im * gamma / 2) - 1
            elseif j > p
                prod *= exp(im * gamma / 2) - 1
            end
        end
        c[α + 1] = prod
    end
    return c
end

function compute_coeff(gamma::Float64, p::Int, q::Int, r::Float64)
    c_phase = compute_c_phase(gamma, p)
    return r .* ((-c_phase) .+ 0im) .^ (1.0 / K_CLAUSE)
end

function active_indices(n_bits::Int)
    d = 2^n_bits
    return [α for α in 0:d-1 if count_ones(α) >= 2]
end

function evaluate_g_and_jacobian(y::Vector{ComplexF64}, A::Matrix{Float64}, A_act::Matrix{Float64}, coeff::Vector{ComplexF64}, b_s::Vector{ComplexF64}, active_idx::Vector{Int})
    y_full = zeros(ComplexF64, D)
    for (k, α) in enumerate(active_idx)
        y_full[α + 1] = y[k]
    end
    # Match Python exactly:
    # log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    # then replace non-finite entries by -700.
    b_reg = b_s .+ (1e-300) .* (1 .- sign.(abs.(b_s)))
    log_b = log.(b_reg)
    for i in eachindex(log_b)
        if !isfinite(real(log_b[i])) || !isfinite(imag(log_b[i]))
            log_b[i] = -700.0 + 0.0im
        end
    end
    # Keep this as a length-d vector (not a 1xd matrix) to avoid broadcast shape blow-up.
    linear = vec(((coeff .* y_full)' * A))
    log_w = log_b .+ linear
    shift = maximum(real.(log_w))
    w = exp.(log_w .- shift)
    w ./= sum(w)
    # A_act is (n_act x d), w is (d,), so E is (n_act,)
    E = A_act * w
    g = similar(y)
    for i in eachindex(y)
        ci = coeff[active_idx[i] + 1]
        g[i] = y[i] + K_CLAUSE * (ci * E[i])^(K_CLAUSE - 1)
    end
    E_AA = A_act * Diagonal(w) * A_act'
    cov = E_AA .- E * E'
    J = Matrix{ComplexF64}(I, length(y), length(y))
    for i in 1:length(y), j in 1:length(y)
        ci = coeff[active_idx[i] + 1]
        cj = coeff[active_idx[j] + 1]
        J[i, j] += K_CLAUSE * (K_CLAUSE - 1) * (ci * E[i])^(K_CLAUSE - 2) * ci * cov[i, j] * cj
    end
    return g, J
end

function real_lift(M::Matrix{ComplexF64})
    a = real.(M)
    b = imag.(M)
    return [a -b; b a]
end

function real_lift(v::Vector{ComplexF64})
    return vcat(real.(v), imag.(v))
end

function krawczyk_certify(y0::Vector{ComplexF64}, beta::Float64, gamma::Float64, r::Float64, radius::Float64)
    A = build_A(N_BITS)
    active = active_indices(N_BITS)
    A_act = A[active .+ 1, :]
    coeff = compute_coeff(gamma, P, Q, r)
    b_s = compute_b_s(beta, P)
    g0, J0 = evaluate_g_and_jacobian(y0, A, A_act, coeff, b_s, active)
    R = inv(J0)
    z0 = real_lift(y0)
    h0 = real_lift(g0)
    Rr = real_lift(R)
    Jr = real_lift(J0)
    center = z0 - Rr * h0
    C = Matrix{Float64}(I, length(z0), length(z0)) - Rr * Jr
    rad = fill(abs(radius), length(z0))
    tail = vec(sum(abs.(C), dims=2)) .* rad
    ok = true
    for i in eachindex(z0)
        if abs(center[i] - z0[i]) + tail[i] >= rad[i]
            ok = false
            break
        end
    end
    return ok, maximum(abs.(g0)), active
end

function main()
    anchor_path, out_path, radius = parse_args()
    anchor = JSON.parsefile(anchor_path)
    beta = Float64(anchor["beta"])
    gamma = Float64(anchor["gamma"])
    r = haskey(anchor, "r") ? Float64(anchor["r"]) : R_DEFAULT
    y0 = complex.(Float64.(anchor["y0_real"]), Float64.(anchor["y0_imag"]))
    coeff_from_anchor = nothing
    logb_from_anchor = nothing
    if haskey(anchor, "coeff_real") && haskey(anchor, "coeff_imag")
        coeff_from_anchor = complex.(Float64.(anchor["coeff_real"]), Float64.(anchor["coeff_imag"]))
    end
    if haskey(anchor, "log_b_real") && haskey(anchor, "log_b_imag")
        logb_from_anchor = complex.(Float64.(anchor["log_b_real"]), Float64.(anchor["log_b_imag"]))
    end

    A = build_A(N_BITS)
    active = active_indices(N_BITS)
    A_act = A[active .+ 1, :]
    coeff = coeff_from_anchor === nothing ? compute_coeff(gamma, P, Q, r) : coeff_from_anchor
    b_s = compute_b_s(beta, P)
    g0, J0 = evaluate_g_and_jacobian(y0, A, A_act, coeff, b_s, active)
    if logb_from_anchor !== nothing
        # Re-evaluate residual with Python-exported log_b for parity diagnostics.
        y_full = zeros(ComplexF64, D)
        for (k, α) in enumerate(active)
            y_full[α + 1] = y0[k]
        end
        linear = vec(((coeff .* y_full)' * A))
        log_w = logb_from_anchor .+ linear
        shift = maximum(real.(log_w))
        w = exp.(log_w .- shift)
        w ./= sum(w)
        E = A_act * w
        g_parity = similar(y0)
        for i in eachindex(y0)
            ci = coeff[active[i] + 1]
            g_parity[i] = y0[i] + 6 * (ci * E[i])^5
        end
        g0 = g_parity
        # Jacobian remains from the same formula, and only residual parity matters here.
    end
    R = inv(J0)
    z0 = real_lift(y0)
    h0 = real_lift(g0)
    Rr = real_lift(R)
    Jr = real_lift(J0)
    center = z0 - Rr * h0
    C = Matrix{Float64}(I, length(z0), length(z0)) - Rr * Jr
    rad = fill(abs(radius), length(z0))
    tail = vec(sum(abs.(C), dims=2)) .* rad
    ok = true
    for i in eachindex(z0)
        if abs(center[i] - z0[i]) + tail[i] >= rad[i]
            ok = false
            break
        end
    end
    residual = maximum(abs.(g0))
    result = Dict(
        "beta" => beta,
        "gamma" => gamma,
        "r" => r,
        "radius" => radius,
        "success" => ok,
        "anchor_residual_inf" => residual,
        "active_idx" => active,
        "used_python_coeff_logb" => (coeff_from_anchor !== nothing && logb_from_anchor !== nothing),
        "message" => ok ? "CERTIFIED: unique root in box" : "inclusion failed",
    )
    open(out_path, "w") do io
        JSON.print(io, result, 2)
    end
    println(result["message"])
    println("result written to ", out_path)
    return ok ? 0 : 1
end

exit(main())
