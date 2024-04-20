using Pkg
Pkg.activate("../alp")
Pkg.instantiate()
using YAML, JLD2
include("../tools/Env.jl")
include("../tools/BaseTypes.jl")
include("../agents/DQN.jl")
# include("DDPG.jl")

@assert length(ARGS) > 0

FILE_PREFIX = ARGS[1]
CONFIG = "../config/$FILE_PREFIX.yml"
PATH_PREFIX = "/media/thornail/SteamLinux/RL-finance-experiments"
OUT_PATH = "$PATH_PREFIX/$FILE_PREFIX.jld2"
c = YAML.load_file(CONFIG)

# PATH_ = "../data/roman_raw.csv"
df = CSV.read(c["path_df"], DataFrame)

Base.Filesystem.mkpath("$PATH_PREFIX/$FILE_PREFIX")

idx_end = min(nrow(df), c["train_end_idx"])

train_env = Env(
    df[c["train_start_idx"] : idx_end, :];
    w_size        = c["trading_step"], 
    commission    = c["commission"], 
    exclude_cols  = c["exclude_cols"],
    need_to_scale = true
)

actions = T.(c["actions"])
actions = [Iterators.product(actions, actions)...]
pushfirst!(actions, (0.0, 0.0))

run = Run([], c)
experiment_len = c["init_train_len"] + c["test_step"] * c["test_num"] + c["test_len"]
for exp_idx in 1:c["n_experiments"]
    dqn = init!(
        in_feats  = length(train_env.feats_for_model),
        out_feats  = length(actions),
        layers    = c["layers"],
        action_space = actions,
        activation = tanh
    )

    start_idx = round(rand() * (nrow(train_env.data) - experiment_len) - 1)
    train_env.start_idx = start_idx
    train_env.end_idx = start_idx + c["init_train_len"]
    train_env.last_point[] = start_idx

    @assert (train_env.start_idx > 0)
    tr_eval, replay_memory = train_dqn(
        dqn,
        train_env;
        episodes=100000,
        max_ep_len=c["max_episode_len"],
        step_=c["window_step"],
        replay_memory_len=c["replay_memory_len"],
        replay_batch=c["replay_batch"],     
        warm_up_episodes = c["warm_up_episodes"],
        alpha = T(c["alpha"]),
        gamma = T(c["gamma"]),
        eval_every = 99999999,
        eval_env = train_env,
        lr_decay = c["lr_decay"],
        merge_every = c["merge_dqns_every"],
        merge_soft  = Bool(c["merge_soft"]),
        merge_alpha  = T(c["merge_alpha"]),
        gradient_clip = c["gradient_clip"],
        eps_start = c["eps_start"],
        eps_end   = c["eps_end"],
        eps_decay = c["eps_decay"],
        rew_decay = c["reward_decay"],
        lr        = c["lr"],
        reg_vol   = T(c["reg_vol"]),
        save_every = -1,
        save_path_pref = "$PATH_PREFIX/$FILE_PREFIX/dqn",
        one_pass = true,
        replay_memory = Tuple{Int, Int, Int, T, Vector{T}, Vector{T}, T}[]
    )

    exper = Experiment(train_env.start_idx, train_env.end_idx, [], dqn.stats["loss"][end], dqn.stats["loss"])

    train_env.start_idx = train_env.end_idx
    train_env.end_idx  += c["trading_step"] + c["window_step"] * c["max_episode_len"]
    println("start of tests... loss of training: $(dqn.stats["loss"][end])")
    for test_idx in 1:c["test_num"]
        train_env.start_idx   += c["test_step"]
        train_env.end_idx     += c["test_step"]
        train_env.last_point[] = train_env.start_idx
        
        train_idx_start = train_env.start_idx
        train_idx_end   = train_env.end_idx

        eval, replay_memory = train_dqn(
            dqn,
            train_env;
            episodes=100000,
            max_ep_len=c["max_episode_len"],
            step_= c["window_step"],
            replay_memory_len=c["replay_memory_len"],
            replay_batch=c["replay_batch"],     
            warm_up_episodes = 0,
            alpha = T(c["alpha"]),
            gamma = T(c["gamma"]),
            eval_every = 99999999,
            eval_env = train_env,
            lr_decay = 0.0,
            merge_every = c["merge_dqns_every"],
            merge_soft  = Bool(c["merge_soft"]),
            merge_alpha  = T(c["merge_alpha"]),
            gradient_clip = c["gradient_clip"],
            eps_start = c["eps_start"],
            eps_end   = c["eps_end"],
            eps_decay = c["eps_decay"],
            rew_decay = c["reward_decay"],
            lr        = dqn.stats["lr"][end],
            reg_vol   = T(c["reg_vol"]),
            save_every = -1,
            save_path_pref = "$PATH_PREFIX/$FILE_PREFIX/dqn",
            one_pass = true,
            replay_memory = replay_memory
        )
        old_start = train_env.start_idx
        old_end   = train_env.end_idx
        train_env.start_idx   = train_env.end_idx
        train_env.end_idx    += c["test_len"]

        move(dqn, false, true)
        res = simulate!(
                train_env, 
                order_action=order_action, 
                step_=c["window_step"], 
                kwargs=dqn,
                clear_every=1)
        res_rand = simulate!(
                train_env, 
                order_action=random_action, 
                step_=c["window_step"], 
                kwargs=dqn,
                clear_every=1)
        res_bk = simulate!(
                train_env, 
                order_action=bk_action, 
                step_=c["window_step"], 
                kwargs=nothing,
                clear_every=1)
        move(dqn, false, false)

        test = Test(sum(res.reward), res.reward, dqn.stats["loss"][end],
                count_orders(res), train_env.start_idx, train_env.end_idx, 
                sum(res_rand.reward), res_rand.reward, count_orders(res_rand),
                sum(res_bk.reward), res_bk.reward, count_orders(res_bk),
                train_idx_start, train_idx_end)
        push!(exper.tests, test)
        println("$(dqn.stats["loss"][end]), $(sum(res.reward)) ,$(sum(res.reward[:20])) | $(res_rand.PnL[]) ,$(sum(res_rand.reward[:20])) | $(res_bk.PnL[]) ,$(sum(res_bk.reward[:20]))")
        train_env.start_idx   = old_start
        train_env.end_idx     = old_end

    end # end of tests
    push!(run.experiments, exper)

    if (exp_idx % c["save_every"]  == 0)
        res = run
        @save "$PATH_PREFIX/$FILE_PREFIX/$(FILE_PREFIX)_$exp_idx.jld2" res
    end

    println("Experiment $exp_idx done!")
end # end of experiments


res = run

@save OUT_PATH res