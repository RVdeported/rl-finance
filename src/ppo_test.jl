using Pkg
Pkg.activate("../alp")
Pkg.instantiate()
using YAML, Plots, Wandb, JLD2
include("./tools/BaseTypes.jl")
include("./tools/Env.jl")
# include("./agents/MRbase.jl")
# include("./agents/AS.jl")
# include("./agents/Grid.jl")
# include("./agents/DQN.jl")
# include("./agents/DDPG.jl")
# include("./agents/TD3.jl")
include("./agents/PPO.jl")
# include("./launch/launch_any.jl")

CONFIG = "../config/dqn_alpha.yml"
c = YAML.load_file(CONFIG)

PATH = c["path_df"]
PATH_ = "../data/roman_raw.csv"
df = CSV.read(PATH, DataFrame)

train_env = Env(
    df[1035003 : 2045003, :],
    "BTCUSDT_FutT";
    w_size       = c["trading_step"], 
    commission   = c["commission"], 
    exclude_cols = c["exclude_cols"],
    need_to_scale=true
)

ppo = init_ppo!(
    in_feats=length(train_env.feats_for_model), 
    P_layers = [128, 64], 
    V_layers = [256, 64], 
    activation = "tanh"
)

train_ppo(
    ppo, train_env,
    episodes    = 150,
    max_ep_len  = 100,
    step_       = 299,
    replay_memory_len = 2048,
    replay_batch = 256,
)