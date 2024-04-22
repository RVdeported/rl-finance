using Distributions

mutable struct ASbase
    alpha::AbstractFloat
    k::AbstractFloat
    gamma::AbstractFloat
    deltaT::AbstractFloat
end

function init_as!(
    a_alpha::AbstractFloat = 1.4,
    a_k    ::AbstractFloat = 1.5,
    a_gamma::AbstractFloat = 0.1,
    a_dT   ::AbstractFloat = 0.001
)
    @assert(a_alpha > 0.0 && a_k >= 0 && a_gamma > 0.0 && a_dT > 0.0 && a_dT < 1.0)
    return ASbase(a_alpha, a_k, a_gamma, a_dT)
end

function quote_as(
    as::ASbase, 
    env::Env, 
    alpha_bias::AbstractFloat = 0.0,
    k_bias    ::AbstractFloat = 0.0,
    gamma_bias::AbstractFloat = 0.0,    
)
    res::Vector{AbstractFloat} = [-1.0, -1.0]

    state  = get_state(env)
    mid_px = state.midprice
    sigma  = state.sigma
    sigma_sqr = sigma * sigma
    qt     = state.Vol

    alpha = as.alpha + alpha_bias
    k     = as.k     + k_bias
    gamma = as.gamma + gamma_bias
    A     = 1 * alpha

    if (alpha <= 0.0 || k <= 0.0 || gamma <= 0.0)
        return res
    end

    res_px = mid_px - qt * gamma * sigma_sqr * (1.0 - as.deltaT)
    spread = gamma * sigma_sqr * (1.0 - as.deltaT) + 2 / gamma * log(1 + gamma / k)
    delta_px = res_px - mid_px
    delta_b  = spread / 2 - delta_px
    delta_a  = spread / 2 + delta_px

    # spread too small - cannot enter
    if (delta_b < 0.0 || delta_a < 0.0)
        return res
    end

    pA = A * exp(-k * delta_a)
    pB = A * exp(-k * delta_b)

    dice1 = rand(Uniform(0.0, 1.0))
    dice2 = rand(Uniform(0.0, 1.0))

    if (pA > dice1) 
        res[1] = mid_px + delta_a
    end
    if (pB > dice2)
        res[2] = mid_px - delta_b
    end

    # if (res[1] > 0.0 || res[2] > 0.0)
    #     print("Entering with pxs $(round(res[2], digits=2))...$(round(res[1], digits=2)) with P $pB...$pA and res_px | px $res_px | $mid_px vol $qt\n")
    # end

    return res
end

function order_action_as(
    env::Env,
    as::ASbase
)
    quotes = quote_as(as, env)

    (quotes[1] > 0.0) && input_order(env, Order(true, 1.0, quotes[1]))
    (quotes[2] > 0.0) && input_order(env, Order(false, 1.0, quotes[2]))
    
end