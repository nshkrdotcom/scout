defmodule ScoutDashboard.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children =
      if Code.ensure_loaded?(ScoutDashboardWeb.Endpoint) do
        [
          {Phoenix.PubSub, name: ScoutDashboard.PubSub},
          ScoutDashboard.TelemetryListener,
          ScoutDashboardWeb.Endpoint
        ]
      else
        # In test/minimal environments the Phoenix endpoint isn't compiled;
        # run a no-op supervisor so the app can boot.
        []
      end

    opts = [strategy: :one_for_one, name: ScoutDashboard.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    endpoint = Module.concat(ScoutDashboardWeb, Endpoint)

    if Code.ensure_loaded?(endpoint) and function_exported?(endpoint, :config_change, 2) do
      apply(endpoint, :config_change, [changed, removed])
    end

    :ok
  end
end
