using Pkg
Pkg.activate("../alp")

using DataFrames, Dates, CSV, StatsBase
include("BaseTypes.jl")

FORMAT = dateformat"yyyy-mm-dd HH:MM:SS.sss"

SCALER_TYPE = UnitRangeTransform

#================================================================#
# Env structure                                                  #
#================================================================#
# structure for holding essencial env data.
# Only float columns considered to be features
mutable struct Env
    data::DataFrame          # dataset holder
    last_point::Ref{Int}     # last index of interest
    w_size::Int              # window of predictions
    real_feats::Vector{Int}  # ids of real features
    feats_for_scale::Vector{Int}
    feats_for_model::Vector{Int}
    date_feats::Vector{Int}  # id of date features in dataset
    commission::Float16      
    lat_ms::Int16            # NOT IMPLEMENTED latency simulation
    orders::Vector{Order}    # holder of currently placed orders
    scaler::Union{SCALER_TYPE, Nothing}
    start_idx::Int
    end_idx::Int
end

# type of float in use
T = Float32

# func for date conversion 
# NB: Julia supports only microsecond level of date precision
global to_dt(x) = DateTime(x[1:23], FORMAT)

#================================================================#
# internal Ctor                                                           #
#================================================================#
# Construct env from input variables. Creates additional features
# noted below in add_features!
# requires at least o_ask_px_1, o_bid_px_1, o_ts_recv and midprice columns
function init_env!(
        df::DataFrame,  
        w_size::Int = 400,
        commission::AbstractFloat = -0.0002,
        lat_ms::Int = 60,
        exclude_cols::Vector{String} = [],
        scaler::Union{SCALER_TYPE, Nothing} = nothing,
        start_idx::Int = -1,
        end_idx::Int = -1
    )
    
    names_ = names(df)
    date_feats = Vector{Int}()
    real_feats = Vector{Int}()
    # we need to look through all columns and distribute them among 
    # real features, date features and non-real values
    for (i, n) in enumerate(eltype.(eachcol(df))) 
        if names_[i] in exclude_cols
            continue
        elseif n <: Int
            push!(date_feats, i)
        elseif n <: AbstractFloat
            push!(real_feats, i)
            df[!,i:i] = convert.(T,df[!, i:i])
        end
    end


    @assert ("o_ask_px_1" in names_) && 
            ("o_bid_px_1" in names_) && 
            ("o_ts_recv"  in names_) &&
            ("midprice"   in names_)
    @assert start_idx <= end_idx
     
    # adding features for trade statistics
    feats_for_scale = deepcopy(real_feats)
    len_original = ncol(df)
    add_features!(df, real_feats)
    isnothing(scaler) && (scaler = fit(SCALER_TYPE, Matrix(df[:, feats_for_scale]), dims=1))
    feats_not_for_scale = symdiff(real_feats, feats_for_scale)
    scaled = StatsBase.transform(scaler, Matrix(df[:, feats_for_scale]))
    scaled = DataFrame(scaled, ["scaled_"*n for n in names(df[1:1, feats_for_scale])])
    scaled = hcat(df, scaled)
    feats_for_model = [len_original+1:ncol(scaled)...]

    start_idx_ = (start_idx > 0) ? start_idx : 1
    end_idx_   = (end_idx > 0) ? end_idx : nrow(df)
    env = Env(scaled, start_idx_, w_size, real_feats, feats_for_scale, feats_for_model, 
            date_feats, commission, lat_ms, [], scaler,
            start_idx_, end_idx_)
    
    return env

end

#------------------------------------------------------------------#
# add_features                                                     #
#------------------------------------------------------------------#
# Addition of features required for performance analysis:
# Vol -- amount of assets owned (can be negative)
# posted_asks, posted_bids -- amount of orders placed but not filled
# PnL -- current calculated PnL (based on portfolio reevaluation)
# Invested -- amount invested into assets, used for portfolio evaluation
function add_features!(df::DataFrame, real_feats::Vector{Int})
    # one more real feature: owned volume
    push!(real_feats, ncol(df) +1)
    df.Vol .= T(0.0)
    push!(real_feats, ncol(df) +1)
    df.absVol .= T(0.0)

    # Posted bids and asks
    push!(real_feats, ncol(df) +1)
    df.posted_asks .= T(0.0)
    push!(real_feats, ncol(df) +1)
    df.posted_bids .= T(0.0)

    # PnL to be calculated
    push!(real_feats, ncol(df) +1)
    df.PnL .= T(0.0)

    # how much we invested into owned assets
    push!(real_feats, ncol(df) +1)
    df.Invested .= T(0.0)    
end

#=================================================================================#
# Ctors                                                                           #
#=================================================================================#
# via DataFrame
Env(df::DataFrame; 
    w_size::Int = 400, 
    commission::AbstractFloat = -0.0002,
    lat_ms::Int = 60,
    exclude_cols::Vector{String} = [],
    scaler::Union{SCALER_TYPE, Nothing} = nothing,
    start_idx::Int = -1,
    end_idx::Int = -1
    ) = init_env!(df, w_size, commission, lat_ms, exclude_cols, scaler, start_idx, end_idx)

# via path to CSV
# Env(path::String; 
#     w_size::Int = 400,
#     commission::AbstractFloat = -0.0002,
#     lat_ms::Int = 60,
#     exclude_cols::Vector{String} = []
#     ) = init_env!(CSV.read(path, DataFrame), w_size, commission, lat_ms, exclude_cols)

# check if we are at the end of dataframe
done(env::Env) = env.w_size > (env.end_idx - env.last_point[] - 1)

# set clear episode - with no financial statistics
function set_up_episode!(
    env::Env, 
    start_point::Int = env.start_idx,
    empty_vol::Bool = true
    )
    @assert start_point < (env.end_idx - env.w_size) "Incorrect start pos $(start_point) with end_idx $(env.end_idx) and w_size $(env.w_size)"
    env.last_point[] = start_point
    empty_vol && (env.data.Vol[start_point] = 0.0)
    empty_vol && (env.data.absVol[start_point] = 0.0)
    empty_vol && (env.data.PnL[start_point] = 0.0)
    empty_vol && (env.data.Invested[start_point] = 0.0)
    empty!(env.orders)
    return !done
end

#---------------------------------------------------------#
# Order registry                                          #
#---------------------------------------------------------#
# Any order put here will be executed in case in intercects
# with best ask/bid. Currently there is no considerations for
# order book depth in this regard
function input_order(env::Env, order::Order)
    order.side_ask ? (get_state(env).posted_asks += 1) :  get_state(env).posted_bids += 1
    push!(env.orders, order)
end

#---------------------------------------------------------#
# Execution of a window                                   #   
#---------------------------------------------------------#
# Once orders are placed, we can simulate outcome of
# trading for a w_size window. Here we determine
# max ask and max bid of the window and check if we can
# match the orders.
# NB: volume is not considered during the current realisation 
struct env_res
    reward::Real        # total reward for a window
    cumm_reward::Real   # reward + PnL value from prev step
    orders_done::Int    # how many orders were executed during the window
    vol_left::Real      # how much assets left
end

function execute!(env::Env, step::Int=2, realize_gain::Bool = true)
    @assert !done(env)

    # getting current item
    item = env.data[env.last_point[], :]
    
    # midprice, max and min asks/bids
    MinA = min(env.data.o_ask_px_1[env.last_point[] : env.last_point[] + env.w_size]...)
    MaxB = max(env.data.o_bid_px_1[env.last_point[] : env.last_point[] + env.w_size]...)
    mid_px = env.data.midprice[env.last_point[] + step]

    #matching the orders
    exRes = match_orders!(orders = env.orders, 
                        best_bid = MaxB, 
                        best_ask = MinA,
                        comm     = env.commission
    )


    # updating of posted order stats
    item.posted_asks -= exRes.executed_asks
    item.posted_bids -= exRes.executed_bids

    @assert (item.posted_asks >= 0) && (item.posted_bids >= 0) "Negative A or B: $(item.posted_asks)/$(item.posted_bids)"

    # updating factually Invested sum and volume 
    item.Invested -= exRes.PnL_delta
    item.Vol += exRes.q_delta

    # estimation of portfolio real value
    real_val =  item.Vol * mid_px

    # recording difference to PnL and updating Invested to real value 
    delta = real_val - item.Invested
    item.PnL += delta
    item.Invested = real_val

    # if set, zero the volume (considered sold since PnL effect 
    # from the transaction already accounted for above)
    realize_gain && (item.Vol      = 0.0)
    realize_gain && (item.Invested = 0.0)

    item.absVol = abs(item.Vol)

    return env_res(delta, item.PnL, exRes.total_executed, item.Vol) 
end



#---------------------------------------------------------#
# Enviroment step                                         #   
#---------------------------------------------------------#
# After we evaluated our actions, we can step-up
# step() will move forward pointer to relevant new
# data copying financial stats data from previous step
function step(env::Env, step::Int=2)
    @assert !done(env)
    @assert step < env.w_size "Step exceeds w_size: $(step) vs $(env.w_size)"

    curr_item = env.data[env.last_point[], :]
    new_item  = env.data[env.last_point[] + step, :]
    new_item.posted_asks = curr_item.posted_asks
    new_item.posted_bids = curr_item.posted_bids
    new_item.PnL = curr_item.PnL
    new_item.Vol = curr_item.Vol
    new_item.absVol = curr_item.absVol
    new_item.Invested = curr_item.Invested

    env.last_point[] += step
    return get_state(env)
end

# getters for current state
function get_state(env::Env, scale::Bool = false)
    scale && (return env.data[env.last_point[], env.feats_for_model])
    return env.data[env.last_point[], env.real_feats]
end

function get_state(env::Env, idx::Int, scale::Bool = false)
    scale && (return env.data[idx, env.feats_for_model])
    return env.data[idx, env.real_feats]
end



#---------------------------------------------------------#
# simulation of enviroment                                #   
#---------------------------------------------------------#
# Function need some Function which accepts Env, state and additional arguments
# and expects it to fill the env with orders for evaluation.
struct sim_result
    reward::Vector{AbstractFloat}
    executed_orders::Vector{Int}
    PnL::Ref{AbstractFloat}
end

function simulate!(
    env::Env;               
    order_action::Function,     # accepts Env, DataFrameRow, Any and trigget input_order(Env, Order)
    step_::Int,
    kwargs,                     # anything needed for order_action()
    clear_env_at_step::Bool = false,
    clear_every::Int = 5
)
    set_up_episode!(env, env.start_idx)
    res_sim = sim_result([], [], 0.0)
    iter = 0
    while !done(env)
        (iter % clear_every == 0) && (set_up_episode!(env, env.last_point[], true))
        # if required, clear all outstanding orders and statistics
        clear_env_at_step && (set_up_episode!(env, env.last_point[]))
        order_action(env, kwargs)
        result = execute!(env, env.w_size)
        push!(res_sim.reward, result.reward)
        push!(res_sim.executed_orders, result.orders_done)
        step(env, step_)
        iter += 1
    end
    res_sim.PnL[] = sum(res_sim.reward)

    return res_sim
end

struct order_exec_stats
    two_completed::Int
    one_completed::Int
    no_action::Int
    other::Int
end

function mm_eval(stats::sim_result)
    two_completed = 0
    one_completed = 0
    no_action = 0
    other = 0
    for n in stats.executed_orders
        (n == 2) && (two_completed += 1)
        (n == 1) && (one_completed += 1)
        (n == 0) && (no_action     += 1)
        (n > 2) &&  (other         += 1)
    end
    return order_exec_stats(two_completed, one_completed, no_action, other)
end