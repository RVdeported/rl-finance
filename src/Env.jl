using Pkg
Pkg.activate("../alp")

using DataFrames, Dates, CSV
include("BaseTypes.jl")

FORMAT = dateformat"yyyy-mm-dd HH:MM:SS.sss"
struct Env
    data::DataFrame
    last_point::Ref{Int}
    w_size::Int
    real_feats::Vector{Int}
    date_feats::Vector{Int}
    commission::Float16  
    lat_ms::Int16
    orders::Vector{Order}
end

T = Float64

global to_dt(x) = DateTime(x[1:23], FORMAT)

function init_env!(
    df::DataFrame, 
    w_size::Int64 = 400,
    commission::AbstractFloat = -0.0002,
    lat_ms::Int = 60,
    exclude_cols::Vector{String} = []
    )
    env = Env(df, 1, w_size, [], [], commission, lat_ms, [])
    names_ = names(env.data)
    for (i, n) in enumerate(eltype.(eachcol(env.data))) 
        if names_[i] in exclude_cols
            i
        elseif n <: Int
            push!(env.date_feats, i)
        elseif n <: AbstractFloat
            push!(env.real_feats, i)
            # df[!, i] = T.(df[!, i])
        end
    end

    @assert ("o_ask_px_1" in names_) && 
            ("o_bid_px_1" in names_) && 
            ("o_ts_recv" in names_) &&
            ("midprice" in names_)

    add_features!(env)
    return env

end

function add_features!(env::Env)
    # one more real feature: owned volume
    push!(env.real_feats, ncol(env.data) +1)
    env.data.Vol .= T(0.0)

    # Posted bids and asks
    push!(env.real_feats, ncol(env.data) + 1)
    env.data.posted_asks .= T(0.0)
    push!(env.real_feats, ncol(env.data) + 1)
    env.data.posted_bids .= T(0.0)

    # PnL to be calculated
    push!(env.real_feats, ncol(env.data) + 1)
    env.data.PnL .= T(0.0)
    
    return env
end

#=================================================================================#
# Ctors                                                                           #
#=================================================================================#
Env(df::DataFrame, 
    w_size::Int = 400, 
    commission::AbstractFloat = -0.0002,
    lat_ms::Int = 60,
    exclude_cols::Vector{String} = []
    ) = init_env!(df, w_size, commission, lat_ms, exclude_cols)
Env(path::String, 
    w_size::Int = 400,
    commission::AbstractFloat = -0.0002,
    lat_ms::Int = 60,
    exclude_cols::Vector{String} = []
    ) = init_env!(CSV.read(path, DataFrame), w_size, commission, lat_ms, exclude_cols)



done(env::Env) = env.w_size > (nrow(env.data) - env.last_point[] - 1)


function set_up_episode!(
    env::Env, 
    start_point::Int = 1,
    empty_vol::Bool = true
    )
    @assert start_point < (nrow(env.data) - env.w_size)
    env.last_point[] = start_point
    empty_vol && (env.data.Vol[start_point] = 0.0)
    empty_vol && (env.data.PnL[start_point] = 0.0)
    empty!(env.orders)
    return !done
end

function input_order(env::Env, order::Order)
    push!(env.orders, order)
end


# XXX: should we account for active bids?
function calc_reward(;
    env::Env,
    vol::AbstractFloat,
    pnl::AbstractFloat,
    end_px::AbstractFloat
)
    pnl += apply_comm(env.commission, vol * end_px, (vol > 0.0))
    return pnl
end

struct env_res
    reward::Real
    orders_done::Int
    vol_left::Real
end

function execute!(env::Env, step::Int=2, realize_gain::Bool = true)
    @assert !done(env)
    item = env.data[env.last_point[], :]
    MaxA = max(env.data.o_ask_px_1[env.last_point[] : env.last_point[] + step]...)
    MinB = min(env.data.o_bid_px_1[env.last_point[] : env.last_point[] + step]...)
    # print("$MaxA, $MinB")

    exRes = match_orders!(orders = env.orders, 
                        best_bid = MinB, 
                        best_ask = MaxA,
                        comm     = env.commission
                        )
    item.Vol += exRes.q_delta
    item.PnL += exRes.PnL_delta
    last_px = item.Vol > 0.0 ? 
                env.data.o_bid_px_1[env.last_point[] + step] : 
                env.data.o_ask_px_1[env.last_point[] + step]
    !realize_gain && (last_px = 0.0)

    # (item.Vol > 0) && println("Left Vol $(item.Vol)")
    reward = calc_reward(
        env    = env, 
        vol    = item.Vol,
        pnl    = item.PnL,
        end_px = last_px
        )
    realize_gain && (item.Vol = 0.0)
    realize_gain && (item.PnL = T(reward))
    return env_res(reward, exRes.total_executed, item.Vol) 
    
end

function step(env::Env, step::Int=2)
    @assert !done(env)
    @assert step < env.w_size

    curr_item = env.data[env.last_point[], :]
    new_item  = env.data[env.last_point[] + step, :]
    new_item.posted_asks = curr_item.posted_asks
    new_item.posted_bids = curr_item.posted_bids
    for n in env.orders
        if n.side_ask
            new_item.posted_asks += 1
        else
            new_item.posted_bids += 1
        end
    end
    new_item.PnL = curr_item.PnL
    new_item.Vol = curr_item.Vol
    env.last_point[] += step
    return get_state(env)
end

function get_state(env::Env)
    return env.data[env.last_point[], env.real_feats]
end

function get_state(env::Env, idx::Int)
    return env.data[idx, env.real_feats]
end


function simulate!(
    env::Env, 
    order_action::Function, 
    step_::Int,
    kwargs
    )
    set_up_episode!(env, 1)
    res_sim = Dict(
        "reward" => [],
        "executed_orders" => []
    )
    while !done(env)
        set_up_episode!(env, env.last_point[])
        state = get_state(env)
        order_action(env, state, kwargs)
        result = execute!(env, step_, true)
        push!(res_sim["reward"], result.reward)
        push!(res_sim["executed_orders"], result.orders_done)

        step(env, step_)
        
    end
    return res_sim
end

