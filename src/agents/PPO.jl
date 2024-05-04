# using Pkg
# Pkg.activate("../alp")
# Pkg.add("CUDA")
using ProgressBars
using Flux, CUDA, DataFrames, Distributions
using StatsBase
using Printf
using Setfield
using JLD2
# include("../tools/Env.jl")
# include("../tools/BaseTypes.jl")

# PPO algorithm with policy - value networks
# policy returns parameters to normal distribution (mean, scale)


function do_z()
    st = 0.01
    Z = T[]
    for i in -5.0:st:5.0
        push!(Z, cdf(Normal(0.0, 1.0), i))
    end
    
    Z = [sum(Z[i+10] - Z[i-10]) for i in 11:990]
    Z = [min(z / i, 10) for (z, i) in zip(Z, 0.1:0.01:9.9)]

    return Z
end

NORMAL_Z_ppo = cu(do_z())

mutable struct PPO <: RLModel
    P_model::Chain
    V_model::Chain
    action_space::Int # number of outputs of type T
    stats::Dict{String, Vector}
    action_type  ::ActionType
    stat_algo    ::Union{Nothing, StatAlgo}    
end

function init_ppo!(;
    in_feats::Int64,
    P_layers::Vector{Int64},
    V_layers::Vector{Int64},
    activation::String  = "relu", # also tanh availabel
    action_space::Int   = 2,    
    action_type::ActionType = spread,
    stat_algo::Union{Nothing, StatAlgo} = nothing,
)   

    # can throw error if activation name invalid
    act_f = get_activation_f(activation)
    P = make_chain(P_layers, in_feats, action_space * 2, act_f)
    V = make_chain(V_layers, in_feats, 1, act_f)
    
    ppo = PPO(
        P, V, action_space,
        Dict(
            "reward"    => [],  
            "loss"      => [],    
            "V_labels"  => [],
            "a_norm"    => [],  
            "vol_left"  => [],
            "lr"        => [],      
        ),
        action_type,
        stat_algo,
    )
    @assert(action_type != OU     || (action_space == 3 && stat_algo isa MRbase))
    @assert(action_type != AS     || (action_space == 3 && stat_algo isa ASbase))
    @assert(action_type != spread || action_space == 2)
    return ppo
end

function move(
    ppo::PPO, 
    P_gpu::Bool = false,
    V_gpu::Bool = false
)
    ppo.P_model   = fmap(P_gpu   ? cu : cpu, ppo.P_model)
    ppo.V_model   = fmap(V_gpu   ? cu : cpu, ppo.V_model)
end


function move_pr(ppo::PPO, gpu::Bool)
    move(ppo, gpu, false)
end

function get_actions(ppo::PPO, state::CuArray)
    logits = cpu(ppo.P_model(state))
    res = Array{T}(undef, ppo.action_space)
    for i in 1:2:ppo.action_space * 2
        sigm = max(0.00001, logits[i+1])
        res[div(i, 2) + 1] = rand(Normal(logits[i], sigm))
    end
    return res, logits
end
                                    # with 2 dimensions!
function get_actions_prob(ppo::PPO, states, actions)
    logits = ppo.P_model.(states)
    
    logprobs = logits2logprob.(logits, actions)
    probs = [exp.(lp) for lp in logprobs]
    
    entropy = sum([-sum.(prob .* logprob) for (prob, logprob) in zip(probs, logprobs)])

    return logprobs, entropy
end

function get_pdf(Z_scores)
    inds = @. Int(min(max(div(Z_scores + 4.9, 0.01), 1), 980))
    Z    = @view NORMAL_Z_ppo[inds]
    bias = cu(fill(T(4.9), length(Z_scores)))
    Z    = @. abs((Z_scores + bias) * Z)
    return Z     
end

function logits2logprob(logits, actions)
    m = @view logits[1:2:end]
    s = @view logits[2:2:end]
    Z = actions .- m 
    Z = Z ./ abs.(s)
    # inds = @. div(Z + 3.0, 0.01) 
    pdf = get_pdf(Z)
    pdf = @. log(pdf)
    # Z = [logpdf(Normal(logits[i * 2 - 1], abs(logits[i * 2])), actions[i]) for i in eachindex(actions)]
    # Z = @. logpdf(Normal(0.0, 1.0), Z)
    # println("NON_FIeNITE $pdf, $Z, $m, $s")
    return pdf
end

function train_ppo(
    ppo::PPO,
    env::Env;
    episodes::Int                   = 100,
    max_ep_len::Int                 = 100,
    step_::Int                      = 100,
    replay_memory_len::Int          = 1024,
    replay_batch::Int               = 128,
    replay_mini_batch::Int          = 32,
    warm_up_episodes::Int           = 100,
    eval_every::Int                 = 100,
    eval_env::Union{Env, Nothing}   = nothing,
    gradient_clip::AbstractFloat    = 1e-3,
    policy_clip::AbstractFloat      = 0.1,
    entropy_loss_c::AbstractFloat   = 0.3,
    value_loss_c::AbstractFloat     = 0.8,
    rew_decay::AbstractFloat        = 0.7,
    reg_vol::T                      = T(0.7),
    reg_action::T                   = T(1e-5),
    rew_offset::T                   = T(0.0),
    lr::AbstractFloat               = 1e-6,
    lr_decay::AbstractFloat         = 0.994,
    save_every::Int                 = -1,
    save_path_pref::String          = "./model",
    wandb_lg::Union{WandbLogger, Nothing} = nothing,
    wandb_pref::String              = ""
)
    @assert replay_memory_len > replay_batch
    @assert replay_batch      > replay_mini_batch
    @assert lr_decay > 0.0

    mkpath(save_path_pref)

                 # state_idx, env_curidx, actions, rew, state, new_state, curr_volume, actions_raw (mean, var)
    replay_memory = Tuple{Int, Int, Vector{T}, T, Vector{T}, Vector{T}, T, Vector{T}}[]
    add_rm(x) = (length(replay_memory) >= replay_memory_len) ? (popfirst!(replay_memory); push!(replay_memory, x)) : push!(replay_memory, x)
    eval_res = []
    optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr))
    # V_optim = Flux.Optimise.Optimiser(ClipValue(gradient_clip), ADAM(lr_V))
    # P_opt_state_val  = Flux.setup(P_optim,  ppo.P_model)
    # V_opt_state_val  = Flux.setup(V_optim,  ppo.V_model)
    global_step = 1

    move(ppo, true, true)

    bar = ProgressBar(1:episodes)
    for ep_idx in bar
        set_up_episode!(env, env.last_point[], true)
        @assert get_state(env).Vol == 0.0
        @assert get_state(env).PnL == 0.0
        
        avg_reward = 0.0
        vol_left = 0.0
        reward_pnl = []
        reward_model = []
        a_norm = T(0.0)
        for pred_idx in 1:max_ep_len
            done(env) && break
            
            state_idx = env.last_point[]
            state_dr = get_state(env, true)
            state_orig = get_state(env, false)

            mid_px = state_orig.midprice
            state = cu(T[state_dr...])
            
            if ep_idx > warm_up_episodes
                actions, actions_raw = get_actions(ppo, state)
            else
                actions = rand(ppo.action_space)
                actions_raw = rand(ppo.action_space * 2)
            end
            curr_norm = StatsBase.norm(actions)
            a_norm += curr_norm

            global_step += 1

            orders = compose_orders(
                model   = ppo,
                actions = T[actions...],
                state   = state_orig,
                mid_px  = mid_px
            )

            for n in orders
                input_order(env, n)
            end
            
            ex_res = execute!(env, step_, pred_idx == max_ep_len)
            
            # first, pure pnl without modificatuins
            reward = ex_res.reward
            push!(reward_pnl, reward)

            # then, reward for a model including volume and action length regularisation
            reward += -reg_vol * state_orig.absVol + rew_offset - reg_action * curr_norm
            push!(reward_model, reward)
            step(env, step_)

            add_rm((state_idx, env.last_point[], actions, 0.0, 
                    deepcopy([state_dr...]), deepcopy([get_state(env, true)...]),
                    ex_res.vol_left, actions_raw))
            (pred_idx == max_ep_len - 1) && (vol_left = get_state(env, true).Vol)
        end
        avg_reward = reward_pnl[end]
        size_rm = length(replay_memory)
        size_rew = length(reward_model)
        for i in 1:size_rew
            rew_cumm = sum([n * rew_decay ^ (t-1) for (t,n) in enumerate(reward_model)])
            @set! replay_memory[size_rm - size_rew + i][4] = T(rew_cumm)
            popfirst!(reward_model)
        end

        set_description(bar, 
            string(@sprintf("Episode %i... avg reward: %.2f, avg vol: %.2f a_norm: %.2f", 
            ep_idx, avg_reward, vol_left, a_norm / max_ep_len)))

        opt_res = optimize(
            env, ppo; 
            replay_memory       = replay_memory,
            optim               = optim,
            batch_len           = replay_batch,
            mini_batch_len      = replay_mini_batch,
            clipping            = T(policy_clip),
            ent_coef            = T(entropy_loss_c),
            val_coef            = T(value_loss_c)
        )

        optim[2].eta *= lr_decay

        push!(ppo.stats["reward"],     avg_reward)
        push!(ppo.stats["loss"],       opt_res["loss"])
        push!(ppo.stats["V_labels"],   opt_res["V_labels"])
        push!(ppo.stats["a_norm"],     a_norm / max_ep_len)
        push!(ppo.stats["vol_left"],   vol_left)
        push!(ppo.stats["lr"],         optim[2].eta)
        !(wandb_lg isa Nothing) && wandb_log_dict(
            wandb_lg, ppo.stats, wandb_pref)
        
        if (ep_idx % eval_every == 0) && !(isnothing(eval_env))
            res = simulate!(eval_env, order_action=order_action, step_=step_, kwargs=ppo)
            push!(eval_res, res)
        end

        if done(env)
            env.last_point[] = rand(1:30)
        end


        if (save_every > 0 && ep_idx % save_every == 0)
            res = Dict("model" => ppo, "eval" => eval_res)
            @save "$(save_path_pref)_$ep_idx.jld2" res
        end

    end

    move(ppo, false, false)
    return eval_res
end



function optimize(
    env::Env,
    ppo::PPO;
    replay_memory::Vector{Tuple{Int, Int, Vector{T}, T, Vector{T}, Vector{T}, T, Vector{T}}},
    batch_len::Int = 128,
    mini_batch_len::Int = 32,
    optim,
    clipping::T  = T(0.02),
    ent_coef::T = T(0.6),
    val_coef::T = T(0.6)
)  
    ent_coef = cu(ent_coef)
    val_coef = cu(val_coef)

    items = Int[rand(1:length(replay_memory)) for _ in 1:batch_len]
    rm_items = replay_memory[items]

    states = [n[5] for n in rm_items]
    states = cu.(states)
    states_actions = [cu(n[3]) for n in rm_items]
    states_V_values  = vcat(cpu.(ppo.V_model.(states))...)

    next_states = [n[6] for n in rm_items]
    next_states = cu.(next_states)

    reward_eval = [n[4] for n in rm_items]

    #============================================================#
    # Advantage calculation:
    # We consider a PPO advantage calculation:
    #   A(t) = -V(0) + discounted_sum(reward) + g^t * V(t)
    # However, since we usually have long episodes (around 100 moves),
    # we consider  g^t * V(t) = eps and do not include into calculations
    # NB: Discounted reward sum was caluculated earlier
    #============================================================#
    advantages  = reward_eval - states_V_values

    # old policy logits (mean1, sigma1, mean2...)
    logits      = [n[8] for n in rm_items]

    params = Flux.params(ppo.P_model, ppo.V_model)
    loss_arr = []
    for start in 1:mini_batch_len:batch_len
        end_ind = start + mini_batch_len - 1

        # removal of inclompete batch - just skip it
        (batch_len - start < mini_batch_len - 1) && break

        # mini-batch composition
        mb_states         = cu.(states[start:end_ind])
        mb_states_actions = cu.(states_actions[start:end_ind])
        mb_advantages     = cu(advantages[start:end_ind])
        mb_logits         = cu.(logits[start:end_ind])
        mb_values         = cu(states_V_values[start:end_ind])
        mb_rewards        = cu(reward_eval[start:end_ind])

        # log probs of actions, size of action space X minibatch len
        mb_logprobs       = cu.(logits2logprob.(mb_logits, mb_states_actions))

        loss, gs = Flux.withgradient(params) do

            # custom function where Distribution.Normal is replaced 
            # with fixed coefficients
            newlogprob, entropy = get_actions_prob(ppo, mb_states, mb_states_actions)
            newlogprob = cu.(newlogprob)
            
            # new value policy evaluation 
            newvalue = mapreduce(permutedims, vcat, ppo.V_model.(mb_states))
            
            # policy loss
            logratio = @. newlogprob - mb_logprobs
            ratio    = cu(exp.(sum.(logratio)))
            pg_loss1 = @. -mb_advantages * ratio
            pg_loss2 = @. -mb_advantages * clamp.(ratio, 1 - clipping, 1 + clipping)
            pg_loss = mean(max.(pg_loss1, pg_loss2))

            # value loss
            v_loss_unclipped  = mean((newvalue - mb_rewards) .^ 2)
            v_clipped         = @. mb_values + clamp(newvalue - mb_values, -clipping, clipping)
            v_loss_clipped    = @. (v_clipped - mb_rewards)^2
            v_loss_max        = max.(v_loss_unclipped, v_loss_clipped)
            v_loss = T(0.5 * mean(v_loss_max))

            # entropy loss
            entropy_loss = T(mean(entropy))
            
            # for good measure - such logs have to be investigated!
            if (!(isfinite(pg_loss) && isfinite(entropy_loss) && isfinite(v_loss)))
                println("Encountered non-finite losses! $pg_loss $entropy_loss $v_loss, ratio: $ratio, $(mean(mb_advantages))")
                return 100.0
            end

            # final loss return -Equation (9) in PPO article
            pg_loss - ent_coef * entropy_loss + val_coef * v_loss
        end
        if loss != 100.0
            Flux.Optimise.update!(optim, params, gs)
        end
        push!(loss_arr, loss)
        
    end

    return Dict(
        "loss" => mean(loss_arr),
        "V_labels" => mean(states_V_values)
    )
end

function order_action(
    env::Env,
    ppo::PPO
)
    state           = get_state(env)
    scaled_state    = get_state(env, true)
    mid_px          = state.midprice
    scaled_state    = cu(T[scaled_state...])
    action_idx      = cpu(get_actions(ppo, scaled_state)[1])
    orders          = compose_orders(
        model   = ppo,
        actions = action_idx,
        state   = state,
        mid_px  = mid_px
    )
    for n in orders
        input_order(env, n)
    end
end

function train!(
    c::Dict,
    ppo::PPO, 
    train_env::Env,
    test_env:: Union{Env,Nothing},
    save_path::String,
    wandb_lg::Union{WandbLogger, Nothing}
)
    eval_res = train_ppo(
        ppo,
        train_env;
        episodes                = c["episodes"],
        max_ep_len              = c["max_episode_len"],
        step_                   = c["window_step"],
        replay_memory_len       = c["replay_memory_len"],
        replay_batch            = c["replay_batch"],
        replay_mini_batch       = c["replay_mini_batch"],
        warm_up_episodes        = c["warm_up_episodes"],
        eval_every              = c["eval_every"],
        eval_env                = test_env,
        gradient_clip           = c["gradient_clip"],
        policy_clip             = c["policy_clip"],
        entropy_loss_c          = c["entropy_loss"],
        value_loss_c            = c["value_loss"],
        rew_decay               = c["reward_decay"],
        reg_vol                 = T(c["reg_vol"]),
        reg_action              = T(c["reg_action"]),
        rew_offset              = T(c["rew_offset"]),
        lr                      = c["lr"],
        lr_decay                = c["lr_decay"],
        save_every              = c["save_every"],
        save_path_pref          = save_path,
        wandb_lg                = wandb_lg,
        wandb_pref              = get_wandb_pref(c["eval_save_path_pref"])
    )

    return eval_res
end