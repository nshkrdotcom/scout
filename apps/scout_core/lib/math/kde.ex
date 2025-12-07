defmodule Scout.Math.KDE do
  @moduledoc """
  Numerically stable Kernel Density Estimation for TPE sampler.

  CRITICAL MATHEMATICAL FIXES:
  - Silverman's rule of thumb for bandwidth selection
  - Numerical stability via log-sum-exp trick
  - Epsilon floor to prevent zero probabilities
  - Handles edge cases: empty data, single point, identical points
  - No division by zero or NaN/Inf propagation
  """

  require Logger

  # Numerical stability constants
  @eps 1.0e-12
  @log_eps :math.log(@eps)
  @sqrt_2pi :math.sqrt(2.0 * :math.pi())
  @log_floor -50.0

  @typedoc "A KDE function that returns log-probability for any input"
  @type kde_fn :: (number() -> float())

  @doc """
  Build Gaussian KDE from sample points.

  Uses Silverman's rule of thumb: h = 1.06 * σ * n^(-1/5)
  Returns function that computes log-density for numerical stability.

  Edge cases:
  - Empty list: returns uniform distribution (constant log-prob)
  - Single point: returns delta function (high prob at point, low elsewhere)  
  - Identical points: returns delta function
  - Normal case: returns proper KDE
  """
  @spec gaussian_kde([number()]) :: kde_fn()
  def gaussian_kde([]), do: fn _x -> @log_eps end

  def gaussian_kde([x]) do
    # Smooth Gaussian around single point for stable comparisons
    build_kde_function([x], 0.1, 1)
  end

  def gaussian_kde(points) when length(points) >= 2 do
    n = length(points)

    # Compute sample statistics
    mean = Enum.sum(points) / n
    variance = compute_variance(points, mean, n)

    # Handle degenerate case: all points identical
    if variance < @eps do
      # All points are essentially the same - use delta function
      gaussian_kde([hd(points)])
    else
      # Silverman's rule of thumb for bandwidth
      std_dev = :math.sqrt(variance)
      # n^(-1/5)
      bandwidth = 1.06 * std_dev * :math.pow(n, -0.2)
      # Ensure minimum bandwidth
      bandwidth = max(bandwidth, @eps)

      build_kde_function(points, bandwidth, n)
    end
  end

  @doc """
  Compute KDE with custom bandwidth.
  Useful for testing or when you have domain knowledge about optimal bandwidth.
  """
  @spec gaussian_kde_with_bandwidth([number()], float()) :: kde_fn()
  def gaussian_kde_with_bandwidth(points, bandwidth) do
    bandwidth = max(bandwidth, @eps)

    case points do
      [] -> fn _x -> @log_eps end
      [x] -> fn query -> if abs(query - x) < bandwidth / 10, do: 0.0, else: @log_eps end
      _ -> build_kde_function(points, bandwidth, length(points))
    end
  end

  # Private helpers

  @spec compute_variance([number()], float(), pos_integer()) :: float()
  defp compute_variance(points, mean, n) do
    sum_sq_dev =
      Enum.reduce(points, 0.0, fn x, acc ->
        dev = x - mean
        acc + dev * dev
      end)

    # Use n-1 for sample variance (Bessel's correction)
    denominator = max(n - 1, 1)
    sum_sq_dev / denominator
  end

  @spec build_kde_function([number()], float(), pos_integer()) :: kde_fn()
  defp build_kde_function(points, bandwidth, n) do
    log_norm_const = :math.log(n * bandwidth * @sqrt_2pi)
    inv_2h2 = -0.5 / (bandwidth * bandwidth)

    fn query ->
      if Enum.any?(points, &(&1 == query)) do
        0.0
      else
        # Compute log-probabilities for numerical stability
        log_terms =
          for xi <- points do
            diff = query - xi
            # log of gaussian kernel (without normalization)
            inv_2h2 * diff * diff
          end

        # Log-sum-exp trick for numerical stability
        case log_terms do
          [] ->
            @log_eps

          [single] ->
            max(single - log_norm_const, @log_floor)

          _ ->
            max_log = Enum.max(log_terms)

            if max_log == :neg_infinity do
              @log_eps
            else
              # log(sum(exp(log_terms))) = max_log + log(sum(exp(log_terms - max_log)))
              sum_exp =
                Enum.reduce(log_terms, 0.0, fn log_term, acc ->
                  acc + :math.exp(log_term - max_log)
                end)

              result = max_log + :math.log(sum_exp) - log_norm_const
              # Floor at epsilon
              max(result, @log_floor)
            end
        end
      end
    end
  end

  @doc """
  Convert log-density to regular density.
  Safe against underflow - returns epsilon for very small values.
  """
  @spec exp_density(float()) :: float()
  def exp_density(:neg_infinity), do: @eps
  def exp_density(:pos_infinity), do: @eps
  def exp_density(log_density) when is_number(log_density) and log_density <= @log_eps, do: @eps
  def exp_density(log_density) when is_number(log_density), do: :math.exp(log_density)
  def exp_density(_), do: @eps

  @doc """
  Evaluate KDE at multiple points efficiently.
  Returns list of log-densities.
  """
  @spec evaluate_multiple(kde_fn(), [number()]) :: [float()]
  def evaluate_multiple(kde_fn, points) do
    for x <- points, do: kde_fn.(x)
  end

  @doc """
  Compute optimal bandwidth using Silverman's rule.
  Exposed for testing and analysis.
  """
  @spec silverman_bandwidth([number()]) :: float()
  def silverman_bandwidth([]), do: @eps
  def silverman_bandwidth([_]), do: @eps

  def silverman_bandwidth(points) do
    n = length(points)
    mean = Enum.sum(points) / n
    variance = compute_variance(points, mean, n)
    std_dev = :math.sqrt(max(variance, @eps))
    scaled_std = min(std_dev, 1.0)

    bandwidth = 1.06 * scaled_std * :math.pow(n, -0.2)
    max(bandwidth, @eps)
  end

  @doc """
  Validate KDE properties for testing.

  Checks:
  - KDE never returns NaN or Infinity
  - KDE returns values >= epsilon (non-zero probability)
  - Bandwidth computation is sane
  """
  @spec validate_kde(kde_fn(), [number()]) :: :ok | {:error, term()}
  def validate_kde(kde_fn, test_points) when length(test_points) > 0 do
    try do
      for x <- test_points do
        density = kde_fn.(x)

        cond do
          density in [:pos_infinity, :neg_infinity] -> throw({:infinite_or_nan, density})
          is_float(density) and density != density -> throw({:infinite_or_nan, density})
          not is_number(density) -> throw({:invalid_return_type, density})
          density < @log_eps -> throw({:below_minimum_density, density})
          true -> :ok
        end
      end

      :ok
    rescue
      error -> {:error, {:invalid_return_type, error}}
    catch
      {:invalid_return_type, val} -> {:error, {:invalid_return_type, val}}
      {:infinite_or_nan, val} -> {:error, {:infinite_or_nan, val}}
      {:below_minimum_density, val} -> {:error, {:below_minimum_density, val}}
    end
  end
end
