# using Pkg
# Pkg.activate("../alp")
using Wandb, DataFrames, Match

EPS = 1e-8

abstract type RLModel end
abstract type StatAlgo end
abstract type Env_p end

@enum ActionType spread OU AS grid

T = Float32

struct Order
    side_ask::Bool
    qt::AbstractFloat
    px::AbstractFloat
end

struct MatchRes
    q_delta::AbstractFloat
    PnL_delta::AbstractFloat
    total_executed::Int
    executed_asks::Int
    executed_bids::Int
end

MatchRes() = MatchRes(0.0, 0.0, 0, 0, 0)
Base.:+(a::MatchRes, b::MatchRes) =
    MatchRes(
        a.q_delta + b.q_delta, 
        a.PnL_delta + b.PnL_delta,
        a.total_executed + b.total_executed,
        a.executed_asks + b.executed_asks,
        b.executed_bids + b.executed_bids)

apply_comm(comm, x, sell) = sell ? (1 - comm) * x : (1 + comm) * x
sqnorm(x) = sum(abs2, x)

function match_orders!(;
    orders::Vector{Order}, 
    best_bid::AbstractFloat,
    best_ask::AbstractFloat,
    comm::AbstractFloat = 0.0
    )
    # @assert best_ask > best_bid
    
    q_delta = 0.0
    PnL_delta = 0.0
    executed_asks = 0
    executed_bids = 0
    mask_matched = zeros(Bool, length(orders))
    for (i, n) in enumerate(orders)
        @assert n.qt >= 0.0 "Negative volume $(n.qt)"
        @assert n.px >= 0.0 "Negative px $(n.px)"
        if (n.side_ask && n.px <= best_bid)
            q_delta   -= n.qt
            PnL_delta += apply_comm(comm, n.qt * n.px, true)
            executed_asks += 1
            mask_matched[i] = true
        elseif (!n.side_ask && n.px >= best_ask)
            executed_bids += 1
            q_delta   += n.qt
            PnL_delta -= apply_comm(comm, n.qt * n.px, false)
            mask_matched[i] = true
        end
    end
    matched = sum(mask_matched)
    mask_matched = [!n for n in mask_matched]
    filter!(p->popfirst!(mask_matched), orders)
    return MatchRes(q_delta, PnL_delta, matched, executed_asks, executed_bids)
end


function make_chain(layers::Vector{Int}, in_feat::Int, end_feat::Int, transition::Function = relu)
    pushfirst!(layers, in_feat)
    first_L = Chain([Dense(layers[i], layers[i+1], transition; init=Flux.glorot_normal(gain=1)) 
                        for i in 1:(length(layers) - 1)])
    final_L = Dense(pop!(layers), end_feat; init=Flux.glorot_normal(gain=1))

    return  Chain(first_L, final_L)
end

add_rm(x) = (length(replay_memory) >= replay_memory_len
            ) ? (popfirst!(replay_memory); push!(replay_memory, x)
            ) : push!(replay_memory, x)

thr(eps_start::AbstractFloat, 
    eps_end::AbstractFloat, 
    eps_decay::Int, 
    steps::Int) = Float32(eps_end) + (eps_start - eps_end) * exp(- T(steps) / eps_decay)


#------------ General function for stat algorithms -------------------------#
function compose_order_stat_algo(model::RLModel, algo::StatAlgo, state::DataFrameRow, bias...)
    orders = []
    quotes = quote_(algo, state::DataFrameRow, bias...) 

    if (abs(quotes[1] - quotes[2]) < EPS)
        return orders
    end

    # TODO: make qty from config
    (quotes[1] > EPS) && push!(orders, Order(true,  1.0, quotes[1]))
    (quotes[2] > EPS) && push!(orders, Order(false, 1.0, quotes[2]))

    # (quotes[1] > EPS || quotes[1] > EPS) && print("Quoting $(quotes[1])...$(quotes[2])\n")

    return orders
end

#-----------------------------------------------------------------------------#
# Compose discrete order variations                                           #
#-----------------------------------------------------------------------------#
function compose_orders(model::RLModel, action_idx::Int, state::DataFrameRow, mid_px::T)
    #TODO: replace with type distribution
    if      model.action_type == spread
        return compose_orders_spread(model, action_idx, state, mid_px)
    else
        return compose_orders_stat(model, action_idx, state, mid_px)
    end
end

function compose_orders_spread(model::RLModel, action_idx::Int, state::DataFrameRow, mid_px::T)
    orders = []
    qt1 = model.action_space[action_idx][1]
    qt2 = model.action_space[action_idx][2]
    

    ((qt1 < 0.0001) && (qt2 < 0.0001)) && return []

    # TODO: make qty from config
    push!(orders, Order(true,  1.0, mid_px * (1.0 + qt1)))
    push!(orders, Order(false, 1.0, mid_px * (1.0 + qt2)))
    return orders
end

function compose_orders_stat(model::RLModel, action_idx::Int, state::DataFrameRow, mid_px::T)
    # here it is asserted that action space is of lenght 3
    ou_bias   = model.action_space[action_idx]
    return compose_order_stat_algo(model, model.stat_algo, state, ou_bias...)
end

#-----------------------------------------------------------------------------#
# Compose continious order variations                                         #
#-----------------------------------------------------------------------------#
function compose_orders(;
    model::RLModel,
    actions::Vector{T},
    state::DataFrameRow,
    mid_px::AbstractFloat,
    assert_both::Bool = false,
    instr_id::Int8 = Int8(1)
)
    args = [model, actions, state, mid_px, assert_both, instr_id]

    if      model.action_type == spread
        return compose_orders_spread(args...)
    else
        return compose_order_stat_algo(model, model.stat_algo, state, actions...)
    end
end

function compose_orders_spread(
    model::RLModel,
    actions::Vector{T},
    state::DataFrameRow,
    mid_px::AbstractFloat,
    assert_both::Bool = false,
    instr_id::Int8 = Int8(1)
)
    orders = []
    sell_viable = (actions[1] > EPS) && (actions[1] < 9.0)
    buy_viable =  (actions[2] > EPS) && (actions[2] < 9.0)

    assert_both && (!sell_viable || !buy_viable) && return []
    sell_viable && push!(orders, Order(true, 1.0,  mid_px * (1 + actions[1] / 10)))
    buy_viable &&  push!(orders, Order(false, 1.0, mid_px * (1 - actions[2] / 10)))

    # sell_viable && println("Selling $(mid_px * (1 + actions[1] / 10))")
    # buy_viable && println("Buying   $(mid_px * (1 - actions[2] / 10))")

    return orders
end

function compose_orders_stat(    
    model::RLModel,
    actions::Vector{T},
    state::DataFrameRow,
    mid_px::AbstractFloat,
    assert_both::Bool = true,
    instr_id::Int8 = Int8(1)
)
    return compose_order_stat_algo(model, model.stat_algo, state, actions...)
end

#--------------------------------------------#
# Other structs                              #
#--------------------------------------------#

struct Test
    PnL::AbstractFloat
    reward::Vector{AbstractFloat}
    loss::AbstractFloat
    trades::Int
    idx_test_start::Int
    idx_test_end::Int
    PnL_base::AbstractFloat
    reward_base::Vector{AbstractFloat}
    trades_base::Int
    PnL_bk::AbstractFloat
    reward_bk::Vector{AbstractFloat}
    trades_bk::Int
    idx_train_start::Int
    idx_train_end::Int
end

mutable struct Experiment
    idx_train_start::Int
    idx_train_end::Int
    tests::Vector{Test}
    loss_init::AbstractFloat
    loss_init_arr::Vector{AbstractFloat}
end

mutable struct Run
    experiments::Vector{Experiment}
    config::Dict
end

function wandb_log_dict(
    wandb_logger::WandbLogger,
    stats::Dict{String, Vector},
    item_prefix::String = ""
)
    report = Dict()
    for key in keys(stats)
        (length(stats[key]) == 0) && return
        report[item_prefix * key] = stats[key][end]
    end

    begin
        lk = ReentrantLock()
        lock(lk)
        try
            Wandb.log(wandb_logger, report)
        finally
            unlock(lk)
        end
    end

    

end

function get_wandb_pref(s::String)
    return s[findall("/", s)[end-1][end]+1:end]
end

function sharpe(input::Vector, base = 65000.0)
    ret = input .* 0.0
    for i in eachindex(ret[2:end])
        ret[i+1] = ret[i] + input[i] / base
    end
    ret[end] / std(input ./ base)
end


function get_activation_f(act_name::String)
    return @match act_name begin
            "relu" => Flux.relu
            "tanh" => Flux.tanh
        end
    
end