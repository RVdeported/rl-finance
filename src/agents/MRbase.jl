using Match

mutable struct MRbase <: StatAlgo
    in_k    ::AbstractFloat
    out_k   ::AbstractFloat
    crit_k  ::AbstractFloat
    vol_pos ::AbstractFloat
    min_pr  ::AbstractFloat
    stats   ::Dict{String, Vector}
    use_VWAP::Bool
end


function init_mr!(
    a_in_k   ::AbstractFloat,
    a_out_k  ::AbstractFloat,
    a_crit_k  ::AbstractFloat,
    a_vol_pos::AbstractFloat,
    min_pr  ::AbstractFloat = 0.0001,
    use_VWAP ::Bool = true
)
    @assert(a_in_k > a_out_k)
    @assert(a_crit_k > a_in_k)
    return MRbase(a_in_k, a_out_k, a_crit_k, a_vol_pos, min_pr, 
                    Dict(
                        "reward" => [],
                        "vol_left" => [],
                        "orders"   => [0, 0]
                    ), 
                    use_VWAP)
end

function init_mr!(c::Dict)
    return init_mr!(c["mr_in_k"], c["mr_out_k"], c["mr_crit_k"], c["mr_pos_vol"],
        c["mr_min_pr"] / 10000, c["mr_use_VWAP"])
end

function quote_(
    mr_params::MRbase, 
    env::Env,
    theta_bias::AbstractFloat = 0.0, 
    kappa_bias::AbstractFloat = 0.0,
    sigma_bias::AbstractFloat = 0.0  
)
    # general params
    state  = get_state(env)
    mid_px = mr_params.use_VWAP ? state.VWAP : state.midprice
    vol    = state.Vol 

    # OU params 
    theta = state.mean  + theta_bias
    kappa = state.kappa + kappa_bias
    sigma = state.sigma + sigma_bias
    # std step

    res::Vector{T} = [T(-1.0), T(-1.0)]

    if (kappa <= 0.0 || sigma <= 0.0)
        return res
    end 

    std = sigma / sqrt(2 * kappa)
    std_from_mp = (mid_px - theta) / std


    # we do not make orders if there are other 
    # order in place
    # if (state.posted_asks != 0 || state.posted_bids != 0)
    #     return res
    # end

    # is std too high? (if so, we are probably in jump)
    if (abs(std_from_mp) > mr_params.crit_k)
        return res
    end

    # should we enter?
    if (abs(std_from_mp) > mr_params.in_k)
        buy = std_from_mp < 0

        qt = buy ? min(mr_params.vol_pos - vol, mr_params.vol_pos) : 
                 - max(-mr_params.vol_pos - vol, -mr_params.vol_pos)
                
        # seems like we already into the position on this side...
        (qt == 0) && return res
        

        col = buy ? env.instr * "_o_ask_px_1" : env.instr * "_o_bid_px_1"
        px = state[col]
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
    pxs = quote_ou(mr_params, env)
    
    if (pxs[1] > 0.0 && pxs[2] > 0.0)
        input_order(env, Order(!buy, mr_params.vol_pos, pxs[1]))
        input_order(env, Order( buy, mr_params.vol_pos, pxs[2]))
    end
    return
end