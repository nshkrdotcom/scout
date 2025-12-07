import Config

# Use in-memory store for tests to avoid external DB dependency
config :scout_core, :store_adapter, Scout.Store.ETS

# Provide a minimal Repo config so Ecto boots if needed, but default to sandbox
config :scout_core, Scout.Repo,
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  database: "scout_test",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: 5

config :scout_core, ecto_repos: [Scout.Repo]

# Quieter test output
config :logger, level: :warning

# Disable HTTP API client for Swoosh in tests to avoid hackney dependency
config :swoosh, :api_client, false

# Dashboard stays off during tests
config :scout_dashboard, :enabled, false
