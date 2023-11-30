using Pkg
Pkg.activate("../alp")


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
        @assert n.qt >= 0.0
        @assert n.px >= 0.0
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
