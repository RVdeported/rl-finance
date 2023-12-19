using Pkg
Pkg.activate("../alp")
EPS = 1e-4

abstract type RLModel end

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

function compose_orders(;
        sell_delta::AbstractFloat, 
        sell_qt::AbstractFloat = 1.0,
        buy_delta::AbstractFloat,
        buy_qt::AbstractFloat = 1.0,
        mid_px::AbstractFloat,
        assert_both::Bool = true
        )
    orders = []
    sell_viable = (sell_delta > EPS)
    buy_viable = (buy_delta > EPS)
    # println("Prices $(sell_delta), $(buy_delta), $(sell_viable), $(buy_viable)")
    assert_both && (!sell_viable || !buy_viable) && return []
    sell_viable && push!(orders, Order(true, sell_qt, mid_px + sell_delta))
    buy_viable && push!(orders, Order(false, buy_qt, max(mid_px - buy_delta, EPS)))
    return orders
end

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
