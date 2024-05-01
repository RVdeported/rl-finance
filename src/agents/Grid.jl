using Distributions

mutable struct Grid <: StatAlgo
    delta    ::AbstractFloat
    target_pr::AbstractFloat
    max_qt   ::AbstractFloat
    prev_px  ::AbstractFloat
end

function init_gr!(
    a_delta    ::AbstractFloat = 1.4,
    a_target_pr::AbstractFloat = 1.5,
    a_max_qt   ::AbstractFloat = 13
)
    @assert(a_delta > 0.0 && a_target_pr >= 0 && a_max_qt > 0)
    return Grid(a_delta, a_target_pr, a_max_qt, -1.0)
end

function init_gr!(c::Dict)
    return init_gr!(c["gr_delta"], c["gr_target_pr"], c["gr_max_qt"])
end

function quote_(
    gr            ::Grid, 
    state         ::DataFrameRow,
    pred_px_delta ::AbstractFloat,
    delta_bias    ::AbstractFloat = 0.0,
    target_pr_bias::AbstractFloat = 0.0,    
)
    res::Vector{AbstractFloat} = [-1.0, -1.0]

    mid_px      = state.midprice
    pred_px     = mid_px         * (1 + pred_px_delta / 10)
    open_px     = state.open
    delta       = gr.delta       * (1 + delta_bias)
    target_pr   = gr.target_pr   * (1 + target_pr_bias)
    vol         = state.Vol

    der_1       = (pred_px - mid_px) / mid_px
    der_2       = pred_px + mid_px - 2 * open_px

    long_cond   = (der_1 >  delta) && (der_2 > 0.0)
    short_cond  = (der_1 < -delta) && (der_2 < 0.0)
    null_cond   = !(long_cond || short_cond)

    if (delta < 0 || target_pr_bias < 0.0 || target_pr_bias > 1.0 || 
        delta_bias < 0.0 || pred_px < 0.0 || null_cond)
        return res
    end

    # check if there is too much qt in stock
    # remove one qt if stock is too much
    if (long_cond && vol > gr.max_qt) ||
       (short_cond && vol < -gr.max_qt)
        return [
            long_cond ? mid_px * 0.995 : -1.0,
           !long_cond ? mid_px * 1.005 : -1.0
        ]
    end

    res = [
        !long_cond ? mid_px : mid_px * (1 + target_pr),
         long_cond ? mid_px : mid_px * (1 - target_pr)
    ]

    return res

end

function order_action(
    env::Env,
    gr ::Grid
)
    # naive price pred
    state = get_state(env)
    pred_px = (state.mean - state.midprice) / state.midprice

    quotes = quote_(gr, get_state(env), pred_px * 10)

    (quotes[1] > 0.0) && input_order(env, Order(true, 1.0, quotes[1]))
    (quotes[2] > 0.0) && input_order(env, Order(false, 1.0, quotes[2]))
    
end