using Distributions
using Plots
using StatsBase

a = 3
b = 4
n = 100
test_data = rand(Beta(a,b), n)

function proposal(prev, parameters = [10,50])
    return rand(InverseGamma(parameters[1],parameters[2]),2)
end

function cond_prob(ϕ, observations)
    return exp(sum(log.(pdf.(Beta(ϕ[1],ϕ[2]), observations))))
end

function alpha_prob(ϕ_curr, ϕ_prev,observations)
    return min(cond_prob(ϕ_curr, observations)/cond_prob(ϕ_prev, observations)
        ,1)
end

function effective_sample_size(x, variance=var(x))
    N = size(x)[1]
    τ_inv = 1 + 2 * autocor(x, [1])[1]
    K = 2
    while K < N - 2
        Δ = autocor(x, [K])[1] + autocor(x, [K + 1])[1]
        if Δ < 0
            break
        else
            τ_inv += 2*Δ
            K += 2
        end
    end
    return N/τ_inv
end


function mh(m, prior_params)
    a_i = ones(m)
    b_i = ones(m)
    num_accepts = 0
    for i = 1:m-1
        prop_ϕ = proposal([a_i[i], b_i[i]],prior_params)
        alpha_i = alpha_prob(prop_ϕ, [a_i[i], b_i[i]], test_data)
        U = rand(Uniform(0,1))
        if alpha_i >= U
            a_i[i+1] = prop_ϕ[1]
            b_i[i+1] = prop_ϕ[2]
            if i > m/2
                num_accepts = num_accepts + 1
            end
        else
            a_i[i+1] = a_i[i]
            b_i[i+1] = b_i[i]
        end
    end
    return num_accepts, a_i, b_i
end

@time accepts, as, bs = mh(20000, [2, 1/8]) # Bad IG params
@time accepts, as, bs = mh(20000, [100, 400]) # Better IG params
pdfs = pdf.(Beta(mean(as[10001:end]), mean(bs[10001:end])), collect(0:0.01:1))
histogram(test_data, normed=true, label = "Training Data")
plot!(collect(0:0.01:1), pdfs, linewidth=2, label = "MH")
plot!(collect(0:0.01:1), pdf.(Beta(3,4), collect(0:0.01:1)), linewidth = 2,
    label = "True")
savefig("mh_001.png")

mean(as[10001:end]), mean(bs[10001:end])
effective_sample_size(as[10001:end])
effective_sample_size(bs[10001:end])
