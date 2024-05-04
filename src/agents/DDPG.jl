# using Pkg
# Pkg.activate("../alp")
# Pkg.add("CUDA")
using ProgressBars
using Flux, CUDA, DataFrames, Distributions
using StatsBase
using Printf
using Setfield
# include("../tools/Env.jl")
# include("../tools/BaseTypes.jl")

mutable struct DDPG <: RLModel
    At_model::Chain
    A_model::Chain
    Ct_model::Chain
    C_model::Chain
    An_model::Chain
    action_space::Int # number of outputs of type T
    stats::Dict{String, Vector}
    action_type  ::ActionType
    stat_algo    ::Union{Nothing, StatAlgo}    
end

function init_ddpg!(;
    in_feats::Int64,
    A_layers::Vector{Int64},
    C_layers::Vector{Int64},
    activation::String  = "relu", # also tanh availabel
    action_space::Int = 2,    
    action_type::ActionType = spread,
    stat_algo::Union{Nothing, StatAlgo} = nothing,
)
    act_f = get_activation_f(activation)
    A = make_chain(A_layers, in_feats, action_space,     act_f)
    C = make_chain(C_layers, in_feats + action_space, 1, act_f)
    
    ddpg = DDPG(
        A,
        deepcopy(A),
        C,
        deepcopy(C),
        deepcopy(A),
        action_space,
        Dict(
            "loss" => [],
            "A_loss" => [],
            "reward" => [],
            "vol_left" => [],
            "noise" => [],
            "lr" => [],
            "C_labels" => [],
            "a_norm" => []
        ),
        action_type,
        stat_algo,
    )

    @assert(action_type != OU     || (action_space == 3 && stat_algo isa MRbase))
    @assert(action_type != AS     || (action_space == 3 && stat_algo isa ASbase))
    @assert(action_type != spread || action_space == 2)

    return ddpg
end

function move(ddpg::DDPG, 
            A_gpu::Bool  = false, 
            At_gpu::Bool = false, 
            C_gpu::Bool  = false, 
            Ct_gpu::Bool = false,
            An_gpu::Bool = false,
            )
    ddpg.At_model = fmap(At_gpu ? cu : cpu, ddpg.At_model)
    ddpg.A_model  = fmap(A_gpu ? cu : cpu,  ddpg.A_model)
    ddpg.Ct_model = fmap(Ct_gpu ? cu : cpu, ddpg.Ct_model)
    ddpg.C_model  = fmap(C_gpu ? cu : cpu,  ddpg.C_model)
    ddpg.An_model  = fmap(An_gpu ? cu : cpu,  ddpg.An_model)
end


function move_pr(ddpg::DDPG, gpu::Bool)
    move(ddpg, gpu, false, false, false, false)
end

function train_ddpg(
    ddpg::DDPG,
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
    eval_env::Union{Env, Nothing} = nothing,
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
    save_path_pref::String = "./model/",
    save_every::Int    = 1000,
    wandb_lg::Union{WandbLogger, Nothing} = nothing,
    wandb_pref::String = ""
)
    mkpath(save_path_pref)

    replay_memory = Tuple{Int, Int, Vector{T}, T, Vector{T}, Vector{T}, T}[]
    add_rm(x) = (length(replay_memory) >= replay_memory_len) ? (popfirst!(replay_memory); push!(replay_memory, x)) : push!(replay_memory, x)
    eval_res = []
    move(ddpg, true, false, true, false)
    A_optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr_A))
    C_optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr_C))
    A_opt_state_val = Flux.setup(A_optim, ddpg.A_model)
    C_opt_state_val = Flux.setup(C_optim, ddpg.C_model)
    global_step = 1

    norm = Normal(noise_mean, noise_sigma)
    bar = ProgressBar(1:episodes)
    for ep_idx in bar
        set_up_episode!(env, env.last_point[], true)
        @assert get_state(env).Vol == 0.0
        @assert get_state(env).PnL == 0.0
        move(ddpg, false, false, false, false)
        
        avg_reward = 0.0
        noise = 0.0
        vol_left = 0.0
        reward_arr = []
        noise = thr(eps_start, eps_end, eps_decay, global_step)
        for (dest, src) in zip(Flux.params([ddpg.An_model]), 
                                Flux.params([ddpg.A_model]))
            dest .= deepcopy(src) .+ rand(norm, size(src)) .* noise
        end
        # println("An params $ep_idx $(Flux.params(ddpg.An_model)[1][1:1, 1:5])")
        # println("C params $ep_idx $(Flux.params(ddpg.C_model)[1][1:1, 1:5])")
        move(ddpg, false, false, false, false, true)
        a_norm = T(0.0)
        for pred_idx in 1:max_ep_len
            done(env) && break
            
            state_idx = env.last_point[]
            state_dr = get_state(env, true)
            state_orig = get_state(env, false)
            mid_px = state_orig.midprice
            state = cu(T[state_dr...])
            
            actions = cpu(ddpg.An_model(state))
            a_norm += StatsBase.norm(actions)
            # noise = thr(eps_start, eps_end, eps_decay, global_step)
            # actions .+= rand(norm, ddpg.action_space) * noise

            global_step += 1

            orders = compose_orders(
                model   = ddpg,
                actions = actions,
                state   = state_orig,
                mid_px  = mid_px)

            for n in orders
                input_order(env, n)
            end
            
            ex_res = execute!(env, step_, pred_idx == max_ep_len)
            reward = ex_res.reward
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

        move(ddpg)
        loss = optimize(env, ddpg; 
            replay_memory = replay_memory,
            batch_len = replay_batch,
            A_optim = A_opt_state_val,
            C_optim = C_opt_state_val,
            warm  = ep_idx < warm_up_episodes,
            alpha = alpha,
            # gamma = gamma,
            reg_vol = reg_vol,
            reg_action = reg_action,
            rew_offset = rew_offset,
            t_beta = t_beta
        )
        A_optim[2].eta *= lr_decay_A
        C_optim[2].eta *= lr_decay_C

        push!(ddpg.stats["reward"], avg_reward)
        push!(ddpg.stats["loss"], loss["C_loss"])
        push!(ddpg.stats["A_loss"], loss["A_loss"])
        push!(ddpg.stats["C_labels"], loss["labels"])
        push!(ddpg.stats["vol_left"], vol_left)
        push!(ddpg.stats["noise"], noise)
        push!(ddpg.stats["lr"], A_optim[2].eta)
        push!(ddpg.stats["a_norm"], a_norm / max_ep_len)
        !(wandb_lg isa Nothing) && wandb_log_dict(
            wandb_lg, ddpg.stats, wandb_pref)
        if (ep_idx % eval_every == 0) && !(isnothing(eval_env))
            move(ddpg, true, false)
            res = simulate!(eval_env, order_action=order_action, step_=step_, kwargs=ddpg)
            push!(eval_res, res)
            move(ddpg, false, false)
        end

        if done(env)
            env.last_point[] = rand(1:30)
        end

        if (save_every > 0 && ep_idx % save_every == 0)
            res = Dict("model" => ddpg, "eval" => eval_res)
            @save "$(save_path_pref)_$ep_idx.jld2" res
        end
    end
    return eval_res
end


function optimize(
    env::Env,
    ddpg::DDPG;
    replay_memory::Vector{Tuple{Int, Int, Vector{T}, T, Vector{T}, Vector{T}, T}},
    batch_len::Int = 128,
    warm::Bool = true,
    alpha::T = T(0.7),
    # gamma::T = T(0.6),
    reg_vol::T = T(0.7),
    rew_offset::T = T(0.0),
    A_optim,
    C_optim,
    t_beta::AbstractFloat = 0.2,
    reg_action::T = T(1e-5)
)  
    @assert t_beta <= 1.0
    @assert alpha <= 1.0

    items = Int[rand(1:length(replay_memory)) for _ in 1:batch_len]
    rm_items = replay_memory[items]
    
    move(ddpg, false, false, true, false)

    states = [n[5] for n in rm_items]
    states = cu.(states)
    states_actions = [cu(n[3]) for n in rm_items]
    states_v_actions = vcat.(states, states_actions)
    states_Q_values = vcat(cpu.(ddpg.C_model.(states_v_actions))...)

    move(ddpg, false, true, false, true)

    next_states = [n[6] for n in rm_items]
    next_states = cu.(next_states)
    next_actions = ddpg.At_model.(next_states)
    next_states_v_actions = vcat.(next_states, next_actions)
    next_states_Q_values = vcat(cpu.(ddpg.Ct_model.(next_states_v_actions))...)

    Vol_id = findall( x -> occursin("Vol", x), names(get_state(env, true)))[1]
    reward_eval = [n[4] - reg_vol * abs(n[6][Vol_id]) + rew_offset # - reg_action * StatsBase.norm(states_actions)
                    for n in rm_items]
    if warm
        C_labels = reward_eval 
    else
        C_labels = [Qv + alpha * (r + Qt - Qv) for 
                    (Qv, r, Qt) in zip(states_Q_values, reward_eval, next_states_Q_values)]
    end

    move(ddpg, true, false, true, false)
        
    C_labels = cu(C_labels)    
    states_v_actions = cu(mapreduce(permutedims, vcat, states_v_actions))
    states_actions   = cu(mapreduce(permutedims, vcat, states_actions))
    states           = cu(mapreduce(permutedims, vcat, states))
    reg_action_cu = cu(T(reg_action))

    C_val, C_grads = Flux.withgradient(ddpg.C_model) do m
        loss = cu(0.0)
        for i in 1:size(states_v_actions)[1]
            q = m(states_v_actions[i, :]) 
            loss += Flux.mse(q, C_labels[i:i])
        end
        loss / size(states_v_actions)[1]
    end

    Flux.update!(C_optim, ddpg.C_model, C_grads[1])
    
    
    A_val, A_grads = Flux.withgradient(ddpg.A_model) do m
        loss = cu([T(0.0)])
        for i in 1:size(states)[1]
            q = m(states[i, :])
            q = ddpg.C_model(vcat(states[i, :], q)) - cu([sum(q .^ 2) * reg_action_cu])
            loss -= q
        end
        sum((loss / size(states)[1]))
    end
    Flux.update!(A_optim, ddpg.A_model, A_grads[1])

    move(ddpg, false, false, false, false)

    for (dest, src) in zip(Flux.params([ddpg.At_model, ddpg.Ct_model]), 
                           Flux.params([ddpg.A_model, ddpg.C_model]))
        dest .= t_beta .* dest  .+ (1-t_beta) .* src
    end
    return Dict("A_loss" => A_val, "C_loss" => C_val, "labels" => mean(cpu(C_labels)))
end

function order_action(
    env::Env,
    ddpg::DDPG
)
    state = get_state(env)
    scaled_state = get_state(env, true)
    mid_px = state.midprice
    scaled_state = cu([scaled_state...])
    action_idx = cpu(ddpg.A_model(scaled_state))
    orders = compose_orders(
        model = ddpg,
        actions = action_idx, 
        state   = state,
        mid_px = mid_px)
    for n in orders
        # display(n)
        input_order(env, n)
    end
    
end


function train!(
    c::Dict,
    ddpg::DDPG, 
    train_env::Env,
    test_env:: Union{Env,Nothing},
    save_path::String,
    wandb_lg::Union{WandbLogger, Nothing} = nothing
)
    eval_res = train_ddpg(
        ddpg,
        train_env;
        episodes=c["episodes"],
        max_ep_len=c["max_episode_len"],
        step_=c["window_step"],
        replay_memory_len=c["replay_memory_len"],
        replay_batch=c["replay_batch"],     
        warm_up_episodes = c["warm_up_episodes"],
        alpha = T(c["alpha"]),
        gamma = T(c["gamma"]),
        eval_every = c["eval_every"],
        eval_env = test_env,
        gradient_clip = c["gradient_clip"],
        eps_start = c["eps_start"],
        eps_end   = c["eps_end"],
        eps_decay = c["eps_decay"],
        rew_decay = c["reward_decay"],
        lr_A      = c["lr_A"],
        lr_C      = c["lr_C"],
        lr_decay_A= c["lr_decay_A"],
        lr_decay_C= c["lr_decay_C"],
        reg_vol   = T(c["reg_vol"]),
        reg_action= T(c["reg_action"]),
        rew_offset= T(c["rew_offset"]),
        noise_sigma = c["noise_sigma"],
        noise_mean = c["noise_mean"],
        t_beta = c["t_beta"],
        save_path_pref = save_path,
        save_every = c["save_every"],
        wandb_lg    = wandb_lg,
        wandb_pref  = get_wandb_pref(c["eval_save_path_pref"])
    )

    return eval_res
end