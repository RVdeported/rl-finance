using Pkg
Pkg.activate("../alp")
using YAML, Plots, JLD2
include("Env.jl")
include("BaseTypes.jl")
include("DQN.jl")
include("DDPG.jl")

@assert length(ARGS) > 0

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

actions = T.(c["actions"])
actions = [Iterators.product(actions, actions)...]
pushfirst!(actions, (0.0, 0.0))

dqn = init!(
    in_feats  = length(train_env.feats_for_model),
    out_feats  = length(actions),
    layers    = c["layers"],
    action_space = actions
)

eval_res = train_dqn(
    dqn,
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
    reg_vol   = T(c["reg_vol"])
)

res = Dict("eval" => eval_res, "model" => dqn, "config" => c)

@save OUT_PATH res