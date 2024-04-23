using Pkg
Pkg.activate("./alp")
Pkg.instantiate()
using YAML, Plots, JLD2, Match
include("../tools/BaseTypes.jl")
include("../tools/Env.jl")
include("../agents/MRbase.jl")
include("../agents/AS.jl")
include("../agents/DQN.jl")
# include("DDPG.jl")

@assert length(ARGS) > 0

FILE_PREFIX = ARGS[1]
CONFIG = "./config/$FILE_PREFIX.yml"
PATH_PREFIX = "/media/ugrek/Backup/rl-finance"
OUT_PATH = "$PATH_PREFIX/$FILE_PREFIX.jld2"
c = YAML.load_file(CONFIG)

# PATH_ = "../data/roman_raw.csv"
df = CSV.read(c["path_df"], DataFrame)

Base.Filesystem.mkpath("$PATH_PREFIX/$FILE_PREFIX")

train_env = Env(
    df[c["train_start_idx"] : c["train_end_idx"], :],
    "BTCUSDT_FutT";
    w_size       = c["trading_step"], 
    commission   = c["commission"], 
    exclude_cols = c["exclude_cols"],
    need_to_scale= c["need_to_scale"]
)
test_env = Env(
    df[c["test_start_idx"] : c["test_end_idx"], :],
    "BTCUSDT_FutT"; 
    w_size       = c["trading_step"], 
    commission   = c["commission"], 
    exclude_cols = c["exclude_cols"],
    need_to_scale = c["need_to_scale"],
    scaler = train_env.scaler
)

#----------------------------------------#
# defining mode -specifics               #
#----------------------------------------#
mode = @match c["mode"] begin
    "spread" => spread
    "ou"     => OU
    "as"     => AS
end

stat_algo = nothing
actions::Vector{Any} = []

if mode == AS
    stat_algo = init_as!(c)
    actions   = [Iterators.product(T.(c["as_alpha_vars"]), T.(c["as_k_vars"]), T.(c["as_gamma_vars"]))...]
elseif mode == OU
    stat_algo = init_mr!(c)
    actions   = [Iterators.product(T.(c["mr_in_k_vars"]), T.(c["mr_out_k_vars"]), T.(c["mr_crit_k_vars"]))...]
elseif mode == spread
    action_base = T.(c["actions"]) ./ 10000.0
    actions     = [Iterators.product(action_base, action_base)...]
    pushfirst!(actions, (0.0, 0.0))
end

@assert(length(actions) > 0)

dqn = init!(
    in_feats     = length(train_env.feats_for_model),
    out_feats    = length(actions),
    layers       = c["layers"],
    action_space = actions,
    stat_algo    = stat_algo,
    action_type  = mode
)

#-----------------------------------------#
# Launch train-eval                       #
#-----------------------------------------#
eval_res = train_dqn(
    dqn,
    train_env;
    episodes         = c["episodes"],
    max_ep_len       = c["max_episode_len"],
    step_            = c["window_step"],
    replay_memory_len= c["replay_memory_len"],
    replay_batch     = c["replay_batch"],     
    warm_up_episodes = c["warm_up_episodes"],
    alpha            = T(c["alpha"]),
    gamma            = T(c["gamma"]),
    eval_every       = c["eval_every"],
    eval_env         = test_env,
    lr_decay         = c["lr_decay"],
    merge_every      = c["merge_dqns_every"],
    merge_soft       = Bool(c["merge_soft"]),
    merge_alpha      = T(c["merge_alpha"]),
    gradient_clip    = c["gradient_clip"],
    eps_start        = c["eps_start"],
    eps_end          = c["eps_end"],
    eps_decay        = c["eps_decay"],
    rew_decay        = c["reward_decay"],
    lr               = c["lr"],
    reg_vol          = T(c["reg_vol"]),
    save_every       = c["save_every"],
    save_path_pref   = PATH_PREFIX,
)

res = Dict("eval" => eval_res, "model" => dqn, "config" => c)

@save OUT_PATH res