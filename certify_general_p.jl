#!/usr/bin/env julia

"""
Krawczyk certification for arbitrary p (q=3 / 8-SAT).

Uses float evaluation at y0 for g0, J0 and R = inv(J0_real).
Encloses the Jacobian J over the real-lifted box with Arb complex balls (Acb)
for y on active indices, then forms C = I - R * J_box (real Arb) and checks
strict inclusion via conservative row bounds:

  |(R*h0)[i]| + sum_j sup|C_ij| * rad_j < rad_i

This upgrades the previous point-Jacobian-only check to interval J_box.
"""

using JSON
using LinearAlgebra
using Arblib

function parse_args()
    anchor = "anchor_p2_uniform.json"
    out = "certification_result_general.json"
    radius = 1e-6
    prec = 256
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
        elseif ARGS[i] == "--prec" && i < length(ARGS)
            prec = parse(Int, ARGS[i + 1])
            i += 2
        else
            i += 1
        end
    end
    return anchor, out, radius, prec
end

function build_A(p::Int)
    n_bits = 2 * p + 1
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

function real_lift(M::Matrix{ComplexF64})
    a = real.(M)
    b = imag.(M)
    return [a -b; b a]
end

function real_lift(v::Vector{ComplexF64})
    return vcat(real.(v), imag.(v))
end

"""Upper bound on |t| for t in a real Arb ball (interval [m-r, m+r])."""
function sup_abs_real_ball(x::Arb)
    m = Float64(Arblib.midref(x))
    r = Float64(Arblib.radref(x))
    return max(abs(m - r), abs(m + r))
end

"""Real scalar `c` as a thin complex ball."""
function acb_real_scalar(c::Float64, prec::Int)
    return Acb(Arb(c, prec = prec), Arb(0.0, prec = prec))
end

function bm24_clause_arity(q::Int)
    return 1 << q
end

function evaluate_g_and_J_float(
    y::Vector{ComplexF64},
    A::Matrix{Float64},
    A_act::Matrix{Float64},
    coeff_full::Vector{ComplexF64},
    log_b::Vector{ComplexF64},
    active_idx::Vector{Int},
    q::Int,
)
    d = size(A, 1)
    n_act = length(active_idx)
    y_full = zeros(ComplexF64, d)
    for (k, α) in enumerate(active_idx)
        y_full[α + 1] = y[k]
    end

    linear = vec(((coeff_full .* y_full)' * A))
    log_w = log_b .+ linear
    shift = maximum(real.(log_w))
    w = exp.(log_w .- shift)
    w ./= sum(w)

    E = A_act * w
    g = similar(y)
    coeff_act = [coeff_full[α + 1] for α in active_idx]
    k = bm24_clause_arity(q)
    for i in 1:n_act
        g[i] = y[i] + k * (coeff_act[i] * E[i])^(k - 1)
    end

    E_AA = A_act * Diagonal(w) * A_act'
    cov = E_AA .- E * E'
    J = Matrix{ComplexF64}(I, n_act, n_act)
    for i in 1:n_act, j in 1:n_act
        J[i, j] += k * (k - 1) * (coeff_act[i] * E[i])^(k - 2) * coeff_act[i] * cov[i, j] * coeff_act[j]
    end
    return g, J
end

"""Thin Acb from complex scalar."""
function acb_const(c::ComplexF64, prec::Int)
    return Acb(Arb(Float64(real(c)), prec = prec), Arb(Float64(imag(c)), prec = prec))
end

"""Complex ball for active component i from real-lift centers and radii."""
function y_ball_from_z(
    z0::Vector{Float64},
    rad::Vector{Float64},
    n_act::Int,
    i::Int,
    prec::Int,
)
    re = Arblib.add_error(Arb(z0[i], prec = prec), Mag(rad[i]))
    im = Arblib.add_error(Arb(z0[n_act + i], prec = prec), Mag(rad[n_act + i]))
    return Acb(re, im)
end

"""
Enclose J (complex n_act × n_act) over the box in y given by z0, rad (real 2n_act).
Returns Matrix{Acb}.
"""
function evaluate_J_interval(
    z0::Vector{Float64},
    rad::Vector{Float64},
    A::Matrix{Float64},
    A_act::Matrix{Float64},
    coeff_full::Vector{ComplexF64},
    log_b::Vector{ComplexF64},
    active_idx::Vector{Int},
    q::Int,
    prec::Int,
)
    d = size(A, 1)
    n_act = length(active_idx)

    coeff_arb = [acb_const(coeff_full[α], prec) for α in 1:d]
    logb_arb = [acb_const(log_b[s], prec) for s in 1:d]

    y_full = Vector{Acb}(undef, d)
    zero_acb = Acb(Arb(0.0, prec = prec), Arb(0.0, prec = prec))
    fill!(y_full, zero_acb)
    for k in 1:n_act
        α = active_idx[k]
        y_full[α + 1] = y_ball_from_z(z0, rad, n_act, k, prec)
    end

    log_w = Vector{Acb}(undef, d)
    mid_re = Vector{Float64}(undef, d)
    for s in 1:d
        acc = logb_arb[s]
        for α in 1:d
            aval = A[α, s]
            if aval != 0.0
                acc = acc + acb_real_scalar(aval, prec) * coeff_arb[α] * y_full[α]
            end
        end
        log_w[s] = acc
        mid_re[s] = Float64(Arblib.midref(real(log_w[s])))
    end

    shift = maximum(mid_re)
    shift_arb = Acb(Arb(shift, prec = prec), Arb(0.0, prec = prec))
    exp_w = [exp(log_w[s] - shift_arb) for s in 1:d]
    Z = sum(exp_w)
    weights = [exp_w[s] / Z for s in 1:d]

    E = Vector{Acb}(undef, n_act)
    for i in 1:n_act
        acc = Acb(Arb(0.0, prec = prec), Arb(0.0, prec = prec))
        for s in 1:d
            aval = A_act[i, s]
            if aval != 0.0
                acc = acc + acb_real_scalar(aval, prec) * weights[s]
            end
        end
        E[i] = acc
    end

    coeff_act = [coeff_arb[active_idx[k] + 1] for k in 1:n_act]

    Cov = Matrix{Acb}(undef, n_act, n_act)
    for i in 1:n_act, j in 1:n_act
        acc = Acb(Arb(0.0, prec = prec), Arb(0.0, prec = prec))
        for s in 1:d
            c = A_act[i, s] * A_act[j, s]
            if c != 0.0
                acc = acc + acb_real_scalar(c, prec) * weights[s]
            end
        end
        Cov[i, j] = acc - E[i] * E[j]
    end

    k = bm24_clause_arity(q)
    k_f = acb_real_scalar(Float64(k), prec)
    km1 = acb_real_scalar(Float64(k - 1), prec)

    J = Matrix{Acb}(undef, n_act, n_act)
    for i in 1:n_act, j in 1:n_act
        ce_i = coeff_act[i] * E[i]
        d_i = k_f * km1 * ce_i^(k - 2)
        J[i, j] = d_i * coeff_act[i] * Cov[i, j] * coeff_act[j]
        if i == j
            J[i, j] = J[i, j] + acb_real_scalar(1.0, prec)
        end
    end
    return J
end

"""Real-lift a complex Acb Jacobian to 2n × 2n real Arb matrix."""
function real_lift_J_acb(J::Matrix{Acb}, prec::Int)
    n = size(J, 1)
    M = Matrix{Arb}(undef, 2n, 2n)
    for i in 1:n, j in 1:n
        a = real(J[i, j])
        b = imag(J[i, j])
        M[i, j] = a
        M[i, j + n] = -b
        M[i + n, j] = b
        M[i + n, j + n] = a
    end
    return M
end

function krawczyk_certify(anchor_data, radius::Float64, prec::Int)
    p = Int(anchor_data["p"])
    q = Int(anchor_data["q"])
    d = Int(anchor_data["d"])
    n_act = Int(anchor_data["n_active"])
    active_idx = Int.(anchor_data["active_idx"])
    y0 = complex.(Float64.(anchor_data["y0_real"]), Float64.(anchor_data["y0_imag"]))
    coeff = complex.(Float64.(anchor_data["coeff_real"]), Float64.(anchor_data["coeff_imag"]))
    log_b = complex.(Float64.(anchor_data["log_b_real"]), Float64.(anchor_data["log_b_imag"]))

    A = build_A(p)
    if size(A, 1) != d
        error("Dimension mismatch: anchor d=$d but build_A gives $(size(A,1))")
    end
    A_act = A[active_idx .+ 1, :]

    g0, J0 = evaluate_g_and_J_float(y0, A, A_act, coeff, log_b, active_idx, q)
    z0 = real_lift(y0)
    h0 = real_lift(g0)
    J0_real = real_lift(J0)
    R = inv(J0_real)

    n_real = 2 * n_act
    rad_vec = zeros(Float64, n_real)
    for i in 1:n_real
        rad_vec[i] = max(abs(radius), abs(radius) * abs(z0[i]))
    end

    # Interval Jacobian over the box (Arb / Acb)
    J_box_c = evaluate_J_interval(z0, rad_vec, A, A_act, coeff, log_b, active_idx, q, prec)
    J_box_real = real_lift_J_acb(J_box_c, prec)

    R_arb = Matrix{Arb}(undef, n_real, n_real)
    for i in 1:n_real, j in 1:n_real
        R_arb[i, j] = Arb(R[i, j], prec = prec)
    end

    I_arb = Matrix{Arb}(undef, n_real, n_real)
    for i in 1:n_real, j in 1:n_real
        I_arb[i, j] = Arb(i == j ? 1.0 : 0.0, prec = prec)
    end

    # C = I - R * J_box (real Arb matrix)
    C = Matrix{Arb}(undef, n_real, n_real)
    for i in 1:n_real, j in 1:n_real
        s = I_arb[i, j]
        for k in 1:n_real
            s = s - R_arb[i, k] * J_box_real[k, j]
        end
        C[i, j] = s
    end

    Rh0 = R * h0
    ok = true
    bad_i = 0
    for i in 1:n_real
        lin_term = 0.0
        for j in 1:n_real
            lin_term += sup_abs_real_ball(C[i, j]) * rad_vec[j]
        end
        if Float64(abs(Rh0[i])) + lin_term >= rad_vec[i]
            ok = false
            bad_i = i
            break
        end
    end

    return Dict(
        "success" => ok,
        "message" =>
            ok ? "CERTIFIED: unique root in box (interval Jacobian)" :
            "inclusion failed at component $(bad_i) of $(n_real) (interval Jacobian)",
        "method" => "interval_jacobian_arb",
        "prec_bits" => prec,
        "p" => p,
        "q" => q,
        "d" => d,
        "n_active" => n_act,
        "radius" => radius,
        "anchor_residual_inf" => maximum(abs.(g0)),
    )
end

function main()
    anchor_path, out_path, radius, prec = parse_args()
    anchor = JSON.parsefile(anchor_path)
    result = krawczyk_certify(anchor, radius, prec)
    open(out_path, "w") do io
        JSON.print(io, result, 2)
    end
    println(result["message"])
    println("result written to ", out_path)
    return result["success"] ? 0 : 1
end

exit(main())
