using Match

mutable struct MRbase <: StatAlgo
    in_k    ::AbstractFloat
    out_k   ::AbstractFloat
    crit_k  ::AbstractFloat
    vol_pos ::AbstractFloat
    min_pr  ::AbstractFloat
    stats   ::Dict{String, Vector}
    use_VWAP::Bool
    kappa_coef::AbstractFloat
end


function init_mr!(
    a_in_k   ::AbstractFloat,
    a_out_k  ::AbstractFloat,
    a_crit_k  ::AbstractFloat,
    a_vol_pos::AbstractFloat,
    min_pr  ::AbstractFloat = 0.0001,
    # use_VWAP ::Bool = false,
    kappa_coef::AbstractFloat = 1.0
)
    @assert(a_in_k > a_out_k)
    @assert(a_crit_k > a_in_k)
    return MRbase(a_in_k, a_out_k, a_crit_k, a_vol_pos, min_pr, 
                    Dict(
                        "reward" => [],
                        "vol_left" => [],
                        "orders"   => [0, 0]
                    ), 
                    false,
                    kappa_coef        
            )
end

function init_mr!(c::Dict)
    return init_mr!(c["mr_in_k"], c["mr_out_k"], c["mr_crit_k"], c["mr_pos_vol"],
        c["mr_min_pr"] / 10000)
end

function quote_(
    mr_params::MRbase, 
    state::DataFrameRow,
    theta_bias::AbstractFloat = 0.0, 
    kappa_bias::AbstractFloat = 0.0,
    sigma_bias::AbstractFloat = 0.0  
)
    # general params
    mid_px = mr_params.use_VWAP ? state.VWAP : state.midprice
    vol    = state.Vol 

    # OU params 
    theta = state.mean  + theta_bias
    kappa = (state.kappa + kappa_bias) * mr_params.kappa_coef
    sigma = state.sigma + sigma_bias
    # std step

    res::Vector{T} = [T(-1.0), T(-1.0)]

    # liquidate stock if we have any
    if abs(vol) > EPS
        buy = vol < 0.0
        return [
            !buy ? T(mid_px) : -1.0,
             buy ? T(mid_px) : -1.0,
        ]
    end

    if (kappa <= 0.0 || sigma <= 0.0)
        return res
    end 

    std = sigma / sqrt(2 * kappa)
    std_from_mp = (mid_px - theta) / std

    # is std too high? (if so, we are probably in jump)
    if (abs(std_from_mp) > mr_params.crit_k)
        return res
    end

    # should we enter?
    if (abs(std_from_mp) > mr_params.in_k)
        buy = std_from_mp < 0
        
        px = mid_px
        dev = std * mr_params.out_k
        px_pass = theta + (buy ? -dev : dev)

        # check if spread too small
        (abs(px - px_pass) / px < mr_params.min_pr) && return res
        
        res[1] =  buy ? px_pass : px
        res[2] = !buy ? px_pass : px

    end

    return res
end

function order_action(env::Env, mr_params::MRbase)
    pxs = quote_(mr_params, get_state(env))
    
    (pxs[1] > 0.0) && input_order(env, Order(true,  mr_params.vol_pos, pxs[1]))
    (pxs[2] > 0.0) && input_order(env, Order(false, mr_params.vol_pos, pxs[2]))
    return
end