# using Pkg
# Pkg.activate("../alp")
# Pkg.add("CUDA")
using ProgressBars
using Flux, CUDA, DataFrames
using StatsBase
using Printf
using Setfield
using JLD2
using Base
# include("../tools/Env.jl")
# include("../tools/BaseTypes.jl")

T = Float32
mutable struct DQN <: RLModel
    target_model ::Ref{Chain}
    predict_model::Ref{Chain}
    action_space ::Vector{Any} # buy and sell offset
    stats        ::Dict{String, Vector}
    action_type  ::ActionType
    mm_ou        ::Union{Nothing, MRbase}    
    mm_as        ::Union{Nothing, ASbase}    
    env          ::Union{Nothing, Env}       
end


function init!(;
    in_feats::Int64,
    out_feats::Int64,
    layers::Vector{Int64},
    action_space::Vector{Any},
    activation::Function = relu,
    action_type::ActionType = spread,
    mm_ou::Union{Nothing, MRbase} = nothing,
    mm_as::Union{Nothing, ASbase} = nothing
)
    model = make_chain(layers, in_feats, out_feats, activation)
    
    @assert(action_type == spread || !isnothing(MRbase))
    @assert(action_type != OU     || (length(action_space[1]) == 3 && !isnothing(mm_ou)))
    @assert(action_type != AS     || (length(action_space[1]) == 3 && !isnothing(mm_as)))
    @assert(action_type != spread || length(action_space[1]) == 2)

    dqn = DQN(
        model,
        deepcopy(model),
        action_space,
        Dict(
            "loss" => [],
            "reward" => [],
            "no_actions" => [],
            "vol_left" => [],
            "randoms" => [],
            "lr" => []
        ),
        action_type,
        mm_ou,
        mm_as,
        nothing
    )
    return dqn
end

function move(dqn::DQN, target_gpu::Bool, predict_gpu::Bool)
    dqn.predict_model[]  = fmap(predict_gpu ? cu : cpu, dqn.predict_model[])
    dqn.target_model[] = fmap(target_gpu ? cu : cpu,  dqn.target_model[])
end

#-----------------------------------------------------------------------------#
# Compose order variations                                                    #
#-----------------------------------------------------------------------------#
function compose_orders(dqn::DQN, action_idx::Int, mid_px::T)
    #TODO: replace with type distribution
    if      dqn.action_type == spread
        return compose_orders_spread(dqn, action_idx, mid_px)
    elseif  dqn.action_type == OU
        return compose_orders_ou(    dqn, action_idx, mid_px)
    elseif  dqn.action_type == AS
        return compose_orders_as(    dqn, action_idx, mid_px)
    end
    @assert(false)
end

function compose_orders_spread(dqn::DQN, action_idx::Int, mid_px::T)
    orders = []
    qt1 = dqn.action_space[action_idx][1]
    qt2 = dqn.action_space[action_idx][2]
    

    ((qt1 < 0.0001) && (qt2 < 0.0001)) && return []

    # TODO: make qty from config
    push!(orders, Order(true,  1.0, mid_px * (1.0 + qt1)))
    push!(orders, Order(false, 1.0, mid_px * (1.0 + qt2)))
    return orders
end

function compose_orders_ou(dqn::DQN, action_idx::Int, mid_px::T)
    orders = []

    # here it is asserted that action space is of lenght 3
    ou_bias   = dqn.action_space[action_idx]
    quotes = quote_ou(dqn.mm_ou, dqn.env, ou_bias...) 

    ((quotes[1] < 0.0001) && (quotes[2] < 0.0001)) && return []

    print("Quoting at $(quotes[1])...$(quotes[2]), action $ou_bias\n")

    # TODO: make qty from config
    push!(orders, Order(true,  1.0, quotes[1]))
    push!(orders, Order(false, 1.0, quotes[2]))
    return orders
end

function compose_orders_as(dqn::DQN, action_idx::Int, mid_px::T)
    orders = []

    # here it is asserted that action space is of lenght 3
    as_bias   = dqn.action_space[action_idx]
    quotes = quote_as(dqn.mm_as, dqn.env, as_bias...) 

    ((quotes[1] < 0.0001) && (quotes[2] < 0.0001)) && return []

    # print("Quoting at $(quotes[1])...$(quotes[2]), action $as_bias\n")

    # TODO: make qty from config
    (quotes[1] > 0.0) && push!(orders, Order(true,  1.0, quotes[1]))
    (quotes[2] > 0.0) && push!(orders, Order(false, 1.0, quotes[2]))
    return orders
end


function train_dqn(
    dqn::DQN,
    env::Env;
    episodes::Int,
    max_ep_len::Int,
    step_::Int,
    replay_memory_len::Int,
    replay_batch::Int,
    warm_up_episodes::Int = 100,
    alpha::T = 0.6,
    gamma::T = 0.7,
    merge_every::Int = 5,
    merge_soft::Bool = true,
    merge_alpha::T = T(0.5),
    eval_every::Int = 100,
    eval_env::Env = nothing,
    lr_decay::AbstractFloat = 0.95,
    gradient_clip::AbstractFloat = 1e-3,
    eps_start::AbstractFloat = 0.05,
    eps_end::AbstractFloat   = 0.90,
    eps_decay::Int = 5000,
    rew_decay::AbstractFloat = 0.7,
    reg_vol::T = T(0.7),
    lr::AbstractFloat = 1e-5,
    save_every = -1,
    save_path_pref = "./dqn",
    one_pass::Bool = false,
    replay_memory::Vector{Tuple{Int, Int, Int, T, Vector{T}, Vector{T}, T}} = Vector{Tuple{Int, Int, Int, T, Vector{T}, Vector{T}, T}}()
)
    add_rm(x) = (length(replay_memory) >= replay_memory_len) ? (popfirst!(replay_memory); push!(replay_memory, x)) : push!(replay_memory, x)
    eval_res = []
    dqn.env = env
    move(dqn, false, true)

    optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr))
    opt_state_val = Flux.setup(optim, dqn.predict_model[])

    global_step = 1

    bar = ProgressBar(1:episodes)
    for ep_idx in bar
        set_up_episode!(env, env.last_point[], true)
        @assert get_state(env).Vol == 0.0
        @assert get_state(env).PnL == 0.0
        move(dqn, false, true)
        
        avg_reward = 0.0
        random_count = 0
        zero_count = 0
        vol_left = 0.0
        reward_arr = []
        for pred_idx in 1:max_ep_len
            done(env) && break
            
            state_idx = env.last_point[]
            state_dr = get_state(env, true)
            state_orig = get_state(env, false)
            mid_px = state_orig.midprice
            state = cu(T[state_dr...])
            
            policy = cpu(dqn.predict_model[](state))
            if rand() > thr(eps_start, eps_end, eps_decay, global_step)
                action_idx = argmax(policy)
            else
                random_count += 1
                policy .-= min(policy...)
                policy /= sum(policy)
                action_idx = sample(1:length(policy), Weights([1.0/length(policy) for _ in 1:length(policy)]), 1)[1]
            end
            global_step += 1
            (action_idx == 1) && (zero_count += 1)

            orders = compose_orders(dqn, action_idx, mid_px)
            
            for n in orders
                input_order(env, n)
            end
            
            ex_res = execute!(env, step_, pred_idx == max_ep_len)
            reward = ex_res.reward
            push!(reward_arr, reward)
            step(env, step_)

            add_rm((state_idx, env.last_point[], action_idx, 0.0, 
                    deepcopy([state_dr...]), deepcopy([get_state(env, true)...]),
                    ex_res.vol_left))
            # avg_reward += reward
            (pred_idx == max_ep_len - 1) && (vol_left = get_state(env, true).Vol)
        end
        avg_reward = reward_arr[end]
        size_rm = length(replay_memory)
        size_rew = length(reward_arr)
        for i in 1:size_rew
            rew_cumm = sum([n * rew_decay ^ (t-1) for (t,n) in enumerate(reward_arr)])
            @set! replay_memory[size_rm - size_rew + i][4] = T(rew_cumm)
            popfirst!(reward_arr)
        end

        set_description(bar, 
            string(@sprintf("Episode %i... randoms: %i, no_action: %i, avg reward: %.2f, avg vol: %.2f, data point: %d", 
            ep_idx, random_count, zero_count, avg_reward, vol_left,  env.last_point[])))

        move(dqn, false, false)
        loss = optimize(env, dqn; 
            replay_memory = replay_memory,
            batch_len = replay_batch,
            optim = opt_state_val,
            warm  = ep_idx < warm_up_episodes,
            alpha = alpha,
            gamma = gamma,
            reg_vol = reg_vol
        )
        optim[2].eta *= lr_decay
        push!(dqn.stats["reward"], avg_reward)
        push!(dqn.stats["loss"], loss)
        push!(dqn.stats["no_actions"], zero_count)
        push!(dqn.stats["vol_left"], vol_left)
        push!(dqn.stats["randoms"], random_count)
        push!(dqn.stats["lr"], optim[2].eta)
        
        if !merge_soft
            (ep_idx % merge_every == 0) && (dqn.target_model[] = deepcopy(dqn.predict_model[]))
        else
            for (dest, src) in zip(Flux.params(dqn.target_model), 
                                    Flux.params(dqn.predict_model))
                dest .= merge_alpha .* dest  .+ (1-merge_alpha) .* src
            end
        end

        if (ep_idx % eval_every == 0) && !(isnothing(eval_env))
            move(dqn, false, true)
            res = simulate!(eval_env, order_action=order_action, step_=step_, kwargs=dqn)
            push!(eval_res, res)
            move(dqn, false, false)
        end

        if done(env)
            one_pass && break
            env.last_point[] = env.start_idx
        end

        if (save_every > 0 && ep_idx % save_every == 0)
            res = Dict("model" => dqn, "eval" => eval_res)
            @save "$(save_path_pref)_$ep_idx.jld2" res
        end
    end
    return eval_res, replay_memory
end


function optimize(
    env::Env,
    dqn::DQN;
    replay_memory::Vector{Tuple{Int, Int, Int, T, Vector{T}, Vector{T}, T}},
    batch_len::Int = 128,
    warm::Bool = true,
    alpha::T = T(0.7),
    gamma::T = T(0.6),
    reg_vol::T = T(0.7),
    optim
)  
    items = Int[rand(1:length(replay_memory)) for _ in 1:batch_len]
    rm_items = replay_memory[items]
    

    move(dqn, false, true)

    states = [n[5] for n in rm_items]
    states = cu.(states)
    states_action_values = dqn.predict_model[].(states)
    states_action_values = [max(cpu(n)...) for n in states_action_values]
    # mapreduce(permutedims, vcat, states_action_values)
    states = mapreduce(permutedims, vcat, cpu(states))

    move(dqn, true, false)

    next_states = [n[6] for n in rm_items]
    next_states = cu.(next_states)
    next_action_values_ = dqn.target_model[].(next_states)
    next_action_values = [max(cpu(n)...) * gamma for n in next_action_values_]
    # mapreduce(permutedims, vcat, next_action_values)
    next_states = nothing

    Vol_id = findall( x -> occursin("Vol", x), names(get_state(env, true)))[1]
    reward_eval = [n[4] - abs(n[6][Vol_id]) * reg_vol for n in rm_items]
    # reward_eval = [log(max(abs(n[4]), 1e-7)) * sign(n[4]) - abs(n[6][Vol_id]) * reg_vol for n in rm_items]
    # reward_eval = [ 10 - abs(n[6][Vol_id]) * reg_vol for n in rm_items]
    if warm
        labels = reward_eval 
    else
        labels = [Qv + alpha * (r + Qt - Qv) for 
                    (Qv, r, Qt) in zip(states_action_values, reward_eval, next_action_values)]
    end
    # println("labels")
    # println(labels)
    # println("values")
    # println(states_action_values)
    # println("next values_")
    # println(next_action_values_)
    # println("next values")
    # println(next_action_values)
    # println("reward eval")
    # println(reward_eval)
    labels = cu(labels)
    

    next_states = nothing

    move(dqn, false, true)
        
    states = cu(states)
    actions_dim = (length(rm_items), length(dqn.action_space))
    actions = [n[3] for n in rm_items]
    action_idx = zeros(Float16, actions_dim)
    [action_idx[i, n] = 1 for (i, n) in zip(1:actions_dim[1], actions)]
    action_idx = cu(action_idx)

    val, grads = Flux.withgradient(dqn.predict_model[]) do m
        loss = cu(0.0)
        for i in 1:size(states)[1]
            res = m(states[i, :])
            res = sum(res .* action_idx[i, :])
            loss += Flux.mse(res, labels[i:i])
        end
        loss / size(states)[1]
    end
    
    # print("Optimizing $(grads[1])\n")
    Flux.update!(optim, dqn.predict_model[], grads[1])

    move(dqn, false, false)
    return val
end

MODEL = nothing
function order_action(
    env::Env,
    dqn::DQN
)
    dqn.env = env
    state = get_state(env)
    scaled_state = get_state(env, true)
    mid_px = state.midprice
    scaled_state = cu([scaled_state...])
    action_idx = argmax(cpu(dqn.predict_model[](scaled_state)))
    orders = compose_orders(dqn, action_idx, mid_px)
    for n in orders
        # display(n)
        input_order(env, n)
    end
    
end

function random_action(
    env::Env,
    dqn::DQN
)
    dqn.env = env
    state = get_state(env)
    mid_px = state.midprice
    action_idx = Int64(round(rand() * (length(dqn.action_space) - 1)) + 1)
    orders = compose_orders(dqn, action_idx, mid_px)
    for n in orders
        # display(n)
        input_order(env, n)
    end
end

