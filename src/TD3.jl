using Pkg
Pkg.activate("../alp")
Pkg.add("CUDA")
using ProgressBars
using Flux, CUDA, DataFrames, Distributions
using StatsBase
using Printf
using Setfield
using JLD2
include("Env.jl")
include("BaseTypes.jl")

mutable struct TD3 <: RLModel
    At_model::Chain
    A_model::Chain
    C1t_model::Chain
    C1_model::Chain
    C2t_model::Chain
    C2_model::Chain
    An_model::Chain
    action_space::Int # number of outputs of type T
    stats::Dict{String, Vector}
end

function init_td3!(;
    in_feats::Int64,
    A_layers::Vector{Int64},
    C_layers::Vector{Int64},
    action_space::Int = 2
)
    A = make_chain(A_layers, in_feats, action_space, Flux.relu)
    C1 = make_chain(C_layers, in_feats + action_space, 1, Flux.relu)
    C2 = make_chain(C_layers, in_feats + action_space, 1, Flux.relu)
    
    td3 = TD3(
        A,
        deepcopy(A),
        C1,
        deepcopy(C1),
        C2,
        deepcopy(C2),
        deepcopy(A),
        action_space,
        Dict(
            "loss" => [],
            "A_loss" => [],
            "reward" => [],
            "no_actions" => [],
            "vol_left" => [],
            "noise" => [],
            "lr" => [],
            "C_labels" => [],
            "a_norm" => []
        )
    )
    return td3
end

function move(td3::TD3, 
            A_gpu::Bool  = false, 
            At_gpu::Bool = false, 
            C1_gpu::Bool  = false, 
            C1t_gpu::Bool = false,
            C2_gpu::Bool  = false, 
            C2t_gpu::Bool = false,
            An_gpu::Bool = false,
            )
    td3.At_model  = fmap(At_gpu  ? cu : cpu, td3.At_model)
    td3.A_model   = fmap(A_gpu   ? cu : cpu, td3.A_model)
    td3.C1t_model = fmap(C1t_gpu ? cu : cpu, td3.C1t_model)
    td3.C1_model  = fmap(C1_gpu  ? cu : cpu, td3.C1_model)
    td3.C2t_model = fmap(C2t_gpu ? cu : cpu, td3.C2t_model)
    td3.C2_model  = fmap(C2_gpu  ? cu : cpu, td3.C2_model)
    td3.An_model  = fmap(An_gpu  ? cu : cpu, td3.An_model)
end



function train_td3(
    td3::TD3,
    env::Env;
    episodes::Int,
    max_ep_len::Int,
    step_::Int,
    replay_memory_len::Int,
    replay_batch::Int,
    warm_up_episodes::Int = 100,
    alpha::T = 0.6,
    gamma::T = 0.7,
    eval_every::Int = 100,
    eval_env::Env = nothing,
    gradient_clip::AbstractFloat = 1e-3,
    eps_start::AbstractFloat = 0.05,
    eps_end::AbstractFloat   = 0.90,
    eps_decay::Int = 5000,
    noise_sigma::AbstractFloat = 1.0,
    noise_mean::AbstractFloat = 2.0,
    rew_decay::AbstractFloat = 0.7,
    reg_vol::T = T(0.7),
    reg_action::T = T(1e-5),
    rew_offset::T = T(0.0),
    lr_A::AbstractFloat = 1e-5,
    lr_C::AbstractFloat = 1e-5,
    lr_decay_A::AbstractFloat = 0.95,
    lr_decay_C::AbstractFloat = 0.95,
    t_beta::AbstractFloat = 0.1,
    save_every::Int = -1,
    save_path_pref::String = "./model",
    reg_weights::T = T(1e-4)
)
    replay_memory = Tuple{Int, Int, Vector{T}, T, Vector{T}, Vector{T}, T}[]
    add_rm(x) = (length(replay_memory) >= replay_memory_len) ? (popfirst!(replay_memory); push!(replay_memory, x)) : push!(replay_memory, x)
    eval_res = []
    move(td3, true, false, true, false, true)
    A_optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr_A))
    C1_optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr_C))
    C2_optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr_C))
    A_opt_state_val  = Flux.setup(A_optim,  td3.A_model)
    C1_opt_state_val = Flux.setup(C1_optim, td3.C1_model)
    C2_opt_state_val = Flux.setup(C2_optim, td3.C2_model)
    global_step = 1

    norm = Normal(noise_mean, noise_sigma)
    bar = ProgressBar(1:episodes)
    for ep_idx in bar
        set_up_episode!(env, env.last_point[], true)
        @assert get_state(env).Vol == 0.0
        @assert get_state(env).PnL == 0.0
        move(td3)
        
        avg_reward = 0.0
        noise = 0.0
        vol_left = 0.0
        reward_arr = []
        noise = thr(eps_start, eps_end, eps_decay, global_step)
        for (dest, src) in zip(Flux.params([td3.An_model]), 
                                Flux.params([td3.A_model]))
            dest .= deepcopy(src) .+ rand(norm, size(src)) .* noise
        end
        # println("An params $ep_idx $(Flux.params(ddpg.An_model)[1][1:1, 1:5])")
        # println("C params $ep_idx $(Flux.params(ddpg.C_model)[1][1:1, 1:5])")
        move(td3, false, false, false, false, false, false, true)
        a_norm = T(0.0)
        for pred_idx in 1:max_ep_len
            done(env) && break
            
            state_idx = env.last_point[]
            state_dr = get_state(env, true)
            state_orig = get_state(env, false)
            mid_px = state_orig.midprice
            state = cu(T[state_dr...])
            
            if ep_idx < warm_up_episodes
                actions = rand(2) .* 0.5
            else
                actions = cpu(td3.An_model(state))
            end
            
            a_norm += StatsBase.norm(actions)
            # noise = thr(eps_start, eps_end, eps_decay, global_step)
            # actions .+= rand(norm, ddpg.action_space) * noise

            global_step += 1

            orders = compose_orders(
                sell_delta = abs(actions[1]), 
                buy_delta  = abs(actions[2]), 
                mid_px = mid_px)

            for n in orders
                input_order(env, n)
            end
            
            ex_res = execute!(env, step_, pred_idx == max_ep_len)
            # reward = ex_res.reward
            reward = (ex_res.orders_done == 2) ? 100 : (ex_res.orders_done == 1) ? -20 : 0
            push!(reward_arr, reward)
            step(env, step_)

            add_rm((state_idx, env.last_point[], actions, 0.0, 
                    deepcopy([state_dr...]), deepcopy([get_state(env, true)...]),
                    ex_res.vol_left))
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
            string(@sprintf("Episode %i... noise: %.2f, avg reward: %.2f, avg vol: %.2f a_norm: %.2f", 
            ep_idx, noise, avg_reward, vol_left, a_norm / max_ep_len)))

        move(td3)
        loss = optimize(env, td3; 
            replay_memory = replay_memory,
            batch_len = replay_batch,
            A_optim = A_opt_state_val,
            C1_optim = C1_opt_state_val,
            C2_optim = C2_opt_state_val,
            warm  = ep_idx < warm_up_episodes,
            alpha = alpha,
            # gamma = gamma,
            reg_vol = reg_vol,
            reg_action = reg_action,
            rew_offset = rew_offset,
            t_beta = t_beta,
            reg_weights = reg_weights
        )
        A_optim[2].eta *= lr_decay_A
        C1_optim[2].eta *= lr_decay_C
        C2_optim[2].eta *= lr_decay_C

        push!(td3.stats["reward"], avg_reward)
        push!(td3.stats["loss"], loss["C_loss"])
        push!(td3.stats["A_loss"], loss["A_loss"])
        push!(td3.stats["C_labels"], loss["labels"])
        push!(td3.stats["vol_left"], vol_left)
        push!(td3.stats["noise"], noise)
        push!(td3.stats["lr"], A_optim[2].eta)
        push!(td3.stats["a_norm"], a_norm / max_ep_len)
        
        if (ep_idx % eval_every == 0) && !(isnothing(eval_env))
            move(td3, true, false)
            res = simulate!(eval_env, order_action=order_action_td3, step_=step_, kwargs=td3)
            push!(eval_res, res)
            move(td3, false, false)
        end

        if done(env)
            env.last_point[] = rand(1:30)
        end

        if (save_every > 0 && ep_idx % save_every == 0)
            res = Dict("model" => td3, "eval" => eval_res)
            @save "$(save_path_pref)_$ep_idx.jld2" res
        end

    end
    return eval_res
end


function optimize(
    env::Env,
    td3::TD3;
    replay_memory::Vector{Tuple{Int, Int, Vector{T}, T, Vector{T}, Vector{T}, T}},
    batch_len::Int = 128,
    warm::Bool = true,
    alpha::T = T(0.7),
    # gamma::T = T(0.6),
    reg_vol::T = T(0.7),
    rew_offset::T = T(0.0),
    A_optim,
    C1_optim,
    C2_optim,
    t_beta::AbstractFloat = 0.2,
    reg_action::T = T(1e-5),
    reg_weights::T = T(1e-1)
)  
    @assert t_beta <= 1.0
    @assert alpha <= 1.0

    items = Int[rand(1:length(replay_memory)) for _ in 1:batch_len]
    rm_items = replay_memory[items]
    
    move(td3, false, false, true, false, true)

    states = [n[5] for n in rm_items]
    states = cu.(states)
    states_actions = [cu(n[3]) for n in rm_items]
    states_v_actions = vcat.(states, states_actions)
    states_Q1_values = vcat(cpu.(td3.C1_model.(states_v_actions))...)
    states_Q2_values = vcat(cpu.(td3.C2_model.(states_v_actions))...)
    states_Q_values = min.(states_Q1_values, states_Q2_values)

    move(td3, false, true, false, true, false, true)

    next_states = [n[6] for n in rm_items]
    next_states = cu.(next_states)
    next_actions = td3.At_model.(next_states)
    next_states_v_actions = vcat.(next_states, next_actions)
    next_states_Q1_values = vcat(cpu.(td3.C1t_model.(next_states_v_actions))...)
    next_states_Q2_values = vcat(cpu.(td3.C2t_model.(next_states_v_actions))...)
    next_states_Q_values = min.(next_states_Q1_values, next_states_Q2_values)

    Vol_id = findall( x -> occursin("absVol", x), names(get_state(env, true)))[1]
    reward_eval = [n[4] - reg_vol * n[6][Vol_id] + rew_offset # - reg_action * StatsBase.norm(states_actions)
                    for n in rm_items]
    if warm
        C_labels = reward_eval 
    else
        C_labels = [Qv + alpha * (r + Qt - Qv) for 
                    (Qv, r, Qt) in zip(states_Q_values, reward_eval, next_states_Q_values)]
    end

    move(td3, true, false, true, false, true)
        
    C_labels = cu(C_labels)    
    states_v_actions = cu(mapreduce(permutedims, vcat, states_v_actions))
    states_actions   = cu(mapreduce(permutedims, vcat, states_actions))
    states           = cu(mapreduce(permutedims, vcat, states))
    reg_action_cu = cu(T(reg_action))

    # println(states_v_actions[:, 1:3])
    # println(C_labels)



    C_val, C_grads = Flux.withgradient(td3.C1_model) do m
        loss = cu(0.0)
        for i in 1:size(states_v_actions)[1]
            q = m(states_v_actions[i, :])
            loss += Flux.mse(q, C_labels[i:i])
        end
        loss / size(states_v_actions)[1]
    end

    Flux.update!(C1_optim, td3.C1_model, C_grads[1])


    C_val, C_grads = Flux.withgradient(td3.C2_model) do m
        loss = cu(0.0)
        # w_reg = sum(sqnorm, Flux.params(m))
        for i in 1:size(states_v_actions)[1]
            q = m(states_v_actions[i, :])
            loss += Flux.mse(q, C_labels[i:i])
        end
        loss / size(states_v_actions)[1]
    end

    Flux.update!(C2_optim, td3.C2_model, C_grads[1])
    
    A_val = 0.0

    if !warm
        A_val, A_grads = Flux.withgradient(td3.A_model) do m
            loss = cu([T(0.0)])
            # w_reg = sum(sqnorm, Flux.params(m))
            for i in 1:size(states)[1]
                q = m(states[i, :])
                q = td3.C1_model(vcat(states[i, :], q)) - cu([sum(q .^ 2) * reg_action_cu]) 
                loss -= q
            end
            sum((loss / size(states)[1]))
        end
        Flux.update!(A_optim, td3.A_model, A_grads[1])
    end

    move(td3)

    for (dest, src) in zip(Flux.params([td3.At_model, td3.C1t_model, td3.C2t_model]), 
                           Flux.params([td3.A_model, td3.C1_model, td3.C2_model]))
        dest .= t_beta .* dest  .+ (1-t_beta) .* src
    end
    return Dict("A_loss" => A_val, "C_loss" => C_val, "labels" => mean(cpu(C_labels)))
end

function order_action_td3(
    env::Env,
    td3::TD3
)
    state = get_state(env)
    scaled_state = get_state(env, true)
    mid_px = state.midprice
    scaled_state = cu([scaled_state...])
    action_idx = cpu(td3.A_model(scaled_state))
    orders = compose_orders(sell_delta = abs(action_idx[1]), 
                            buy_delta = abs(action_idx[2]), 
                            mid_px = mid_px)
    for n in orders
        # display(n)
        input_order(env, n)
    end
    
end