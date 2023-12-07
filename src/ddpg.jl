using Pkg
Pkg.activate("../alp")
using YAML, Plots, JLD2
include("Env.jl")
include("BaseTypes.jl")
include("DQN.jl")
include("DDPG.jl")

@assert length(ARGS) > 0

# warm_start = false
# (length(ARGS) > 1) && (warm_start = Bool(ARGS[2]))

FILE_PREFIX = ARGS[1]
CONFIG = "../config/$FILE_PREFIX.yml"
OUT_PATH = "../experiments/$FILE_PREFIX.jld2"
c = YAML.load_file(CONFIG)

PATH = c["path_df"]
# PATH_ = "../data/roman_raw.csv"
df = CSV.read(PATH, DataFrame)

train_env = Env(
    df[c["train_start_idx"] : c["train_end_idx"], :]; 
    w_size       = c["trading_step"], 
    commission   = c["commission"], 
    exclude_cols = c["exclude_cols"],
)
test_env = Env(
    df[c["test_start_idx"] : end, :]; 
    w_size       = c["trading_step"], 
    commission   = c["commission"], 
    exclude_cols = c["exclude_cols"],
    scaler = train_env.scaler
)


# if !warm_start
ddpg = init_ddpg!(
    in_feats = length(train_env.feats_for_model),
    A_layers = c["A_layers"],
    C_layers = c["C_layers"],
    action_space = 2
)
# else
#     @load OUT_PATH
#     ddpg = res["model"]
# end

eval_res = train_ddpg(
    ddpg,
    train_env;
    episodes=c["episodes"],
    # episodes=5000,
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
    t_beta = c["t_beta"]
)

res = Dict("eval" => eval_res, "model" => ddpg, "config" => c)

@save OUT_PATH res