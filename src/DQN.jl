using Pkg
Pkg.activate("../alp")
Pkg.add("CUDA")
using ProgressBars
using Flux, CUDA, DataFrames
using StatsBase
using Printf
using Setfield
include("Env.jl")
include("BaseTypes.jl")

T = Float32
struct DQN
    target_model::Ref{Chain}
    predict_model::Ref{Chain}
    action_space::Vector{Tuple{T, T}} # buy and sell offset
    eps_start::AbstractFloat
    eps_end::AbstractFloat
    eps_decay::Int
    rew_decay::AbstractFloat
    reg_vol::AbstractFloat
    lr::AbstractFloat
    stats::Dict{String, Vector}
end


function init!(;
    in_feats::Int64,
    layers::Vector{Int64},
    action_space::Vector{Tuple{T, T}},
    eps_start::AbstractFloat = 0.9,
    eps_end::AbstractFloat   = 0.05,
    eps_decay::Int64         = 10000,
    rew_decay::AbstractFloat = 0.9,
    reg_vol::AbstractFloat   = 15.0,
    lr::AbstractFloat        = 1e-5
)
    pushfirst!(layers, in_feats)
    first_L = Chain([Dense(layers[i], layers[i+1], relu; init=Flux.glorot_normal(gain=1)) 
                        for i in 1:(length(layers) - 1)])
    final_L = Dense(pop!(layers), length(action_space); init=Flux.glorot_normal(gain=1))
    
    model = Chain(first_L, final_L)
    
    dqn = DQN(
        model,
        deepcopy(model),
        action_space,
        eps_start,
        eps_end,
        eps_decay,
        rew_decay,
        reg_vol,
        lr,
        Dict(
            "loss" => [],
            "reward" => [],
            "no_actions" => [],
            "vol_left" => [],
            "randoms" => [],
            "lr" => []
        )
    )
    return dqn
end

thr(dqn::DQN, steps::Int) = dqn.eps_end + (dqn.eps_start - dqn.eps_end) * exp(- T(steps) / dqn.eps_decay)

function move(dqn::DQN, target_gpu::Bool, predict_gpu::Bool)
    dqn.predict_model[]  = fmap(predict_gpu ? cu : cpu, dqn.predict_model[])
    dqn.target_model[] = fmap(target_gpu ? cu : cpu,  dqn.target_model[])
end

function compose_orders(dqn::DQN, action_idx::Int, mid_px::T)
    orders = []
    qt1 = dqn.action_space[action_idx][1]
    qt2 = dqn.action_space[action_idx][2]
    

    ((qt1 < 0.0001) && (qt2 < 0.0001)) && return []

    push!(orders, Order(true, 1.0, mid_px + qt1))
    push!(orders, Order(false, 1.0, mid_px - qt2))
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
    eval_every::Int = 100,
    eval_env::Env = nothing,
    lr_decay::AbstractFloat = 0.95,
    gradient_clip::AbstractFloat = 1e-3
)
    warm_up = true

    replay_memory = Tuple{Int, Int, Int, T, Vector{T}, Vector{T}, T}[]
    add_rm(x) = (length(replay_memory) >= replay_memory_len) ? (popfirst!(replay_memory); push!(replay_memory, x)) : push!(replay_memory, x)
    eval_res = []

    # optim = Adam(dqn.lr)
    optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(dqn.lr))
    
    move(dqn, false, true)
    opt_state_val = Flux.setup(optim, dqn.predict_model[])
    global_step = 1

    bar = ProgressBar(1:episodes)
    for ep_idx in bar
        set_up_episode!(env, env.last_point[] + 1, true)
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
            if rand() > thr(dqn, global_step)
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
            rew_cumm = sum([n * dqn.rew_decay ^ (t-1) for (t,n) in enumerate(reward_arr)])
            @set! replay_memory[size_rm - size_rew + i][4] = T(rew_cumm)
            popfirst!(reward_arr)
        end

        set_description(bar, 
            string(@sprintf("Episode %i... randoms: %i, no_action: %i, avg reward: %.2f, avg vol: %.2f", 
            ep_idx, random_count, zero_count, avg_reward, vol_left)))

        move(dqn, false, false)
        loss = optimize(env, dqn; 
            replay_memory = replay_memory,
            batch_len = replay_batch,
            optim = opt_state_val,
            warm  = ep_idx < warm_up_episodes,
            alpha = alpha,
            gamma = gamma
        )
        optim[2].eta *= lr_decay
        push!(dqn.stats["reward"], avg_reward)
        push!(dqn.stats["loss"], loss)
        push!(dqn.stats["no_actions"], zero_count)
        push!(dqn.stats["vol_left"], vol_left)
        push!(dqn.stats["randoms"], random_count)
        push!(dqn.stats["lr"], optim[2].eta)
        
        (ep_idx % merge_every == 0) && (dqn.target_model[] = deepcopy(dqn.predict_model[]))

        if (ep_idx % eval_every == 0) && !(isnothing(eval_env))
            move(dqn, false, true)
            res = simulate!(eval_env, order_action=order_action, step_=step_, kwargs=dqn)
            push!(eval_res, res)
            move(dqn, false, false)
        end

        if done(env)
            env.last_point[] = 1
        end
    end
    return eval_res
end


function optimize(
    env::Env,
    dqn::DQN;
    replay_memory::Vector{Tuple{Int, Int, Int, T, Vector{T}, Vector{T}, T}},
    batch_len::Int = 128,
    warm::Bool = true,
    alpha::T = T(0.7),
    gamma::T = T(0.6),
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
    # reward_eval = [n[4] - abs(n[6][Vol_id]) * dqn.reg_vol for n in rm_items]
    reward_eval = [log(max(abs(n[4]), 1e-7)) * sign(n[4]) - abs(n[6][Vol_id]) * dqn.reg_vol for n in rm_items]
    # reward_eval = [ 10 - abs(n[6][Vol_id]) * dqn.reg_vol for n in rm_items]
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

